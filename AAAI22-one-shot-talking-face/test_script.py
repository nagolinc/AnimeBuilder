import os
import numpy as np
import torch
import yaml
from models.generator import OcclusionAwareGenerator
from models.keypoint_detector import KPDetector
import argparse
import imageio
from models.util import draw_annotation_box
from models.transformer import Audio2kpTransformer
from scipy.io import wavfile
from tools.interface import read_img,get_img_pose,get_pose_from_audio,get_audio_feature_from_audio,\
    parse_phoneme_file,load_ckpt
import config
import time

import pickle

from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import Dataset, DataLoader

import subprocess


_kp_detector = None
_generator = None
_ph2kp = None

def normalize_kp(kp_source, kp_driving, kp_driving_initial,
                 use_relative_movement=True, use_relative_jacobian=True):

    kp_new = {k: v for k, v in kp_driving.items()}
    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        # kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

class AudioPoseDataset(torch.utils.data.Dataset):
    def __init__(self, frames, audio_seq, ph_seq, rot_seq, trans_seq, opt):
        self.frames = frames
        self.audio_seq = audio_seq
        self.ph_seq = ph_seq
        self.rot_seq = rot_seq
        self.trans_seq = trans_seq
        self.opt = opt

    def __len__(self):
        return self.frames

    def __getitem__(self, rid):
        pad = np.zeros((4, self.audio_seq.shape[1]), dtype=np.float32)
        ph = []
        audio = []
        pose = []
        for i in range(rid - self.opt.num_w, rid + self.opt.num_w + 1):
            if i < 0:
                rot = self.rot_seq[0]
                trans = self.trans_seq[0]
                ph.append(31)
                audio.append(pad)
            elif i >= self.frames:
                ph.append(31)
                rot = self.rot_seq[self.frames - 1]
                trans = self.trans_seq[self.frames - 1]
                audio.append(pad)
            else:
                ph.append(self.ph_seq[i])
                rot = self.rot_seq[i]
                trans = self.trans_seq[i]
                audio.append(self.audio_seq[i * 4:i * 4 + 4])
            tmp_pose = np.zeros([256, 256])
            draw_annotation_box(tmp_pose, np.array(rot), np.array(trans))
            pose.append(tmp_pose)
        #return torch.tensor(ph), torch.tensor(audio), torch.tensor(pose)
        #return torch.tensor(ph), torch.stack([torch.tensor(a) for a in audio]), torch.stack([torch.tensor(p) for p in pose])
        ph=np.array([x for x in ph]).squeeze()
        audio=np.array([np.array(x) for x in audio]).squeeze()
        pose=np.array([np.array(x) for x in pose]).squeeze()
        return ph, audio, pose


def test_with_input_audio_and_image(img_path, audio_path,phs, generator_ckpt, audio2pose_ckpt, save_dir="samples/results",config_prefix="./",decimate=1):

    startTime=time.time()
    
    print("STARTING",time.time()-startTime)
    
    with open(os.path.join(config_prefix,"config_file/vox-256.yaml")) as f:
        config = yaml.safe_load(f)

    #print("GOT config",config)

    # temp_audio = audio_path
    # print(audio_path)
    cur_path = os.getcwd()

    print("READING AUDIO",time.time()-startTime)

    sr,_ = wavfile.read(audio_path)
    if sr!=16000:
        temp_audio = os.path.join(cur_path,"samples","temp.wav")
        #command = "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, temp_audio)
        #os.system(command)
        cmd = ['ffmpeg', '-y', '-i', audio_path, '-async', '1', '-ac', '1', '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', temp_audio]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        temp_audio = audio_path

    print("Loading namespace",time.time()-startTime)

    opt = argparse.Namespace(**yaml.safe_load(open(os.path.join(config_prefix,"config_file/audio2kp.yaml"))))

    print("READING IMAGE",time.time()-startTime)

    img = read_img(img_path).cuda()

    print("GETTING POSE",time.time()-startTime)

    #pickle pose to hash of img_path and reuse if it exists
    pickle_path = os.path.join(save_dir,"pose_"+os.path.basename(img_path)+".pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path,"rb") as f:
            first_pose = pickle.load(f)
    else:
        first_pose = get_img_pose(img_path,processor = "E:\img\openface\OpenFace_2.2.0_win_x64\FeatureExtraction.exe")#.cuda()
        with open(pickle_path,"wb") as f:
            pickle.dump(first_pose,f)

    print("GOT POSE",time.time()-startTime)

    audio_feature = get_audio_feature_from_audio(temp_audio)
    frames = len(audio_feature) // 4
    frames = min(frames,len(phs["phone_list"]))

    print("GOT AUDIO FEATURE, CONVERTING TO POSE",time.time()-startTime)

    tp = np.zeros([256, 256], dtype=np.float32)
    draw_annotation_box(tp, first_pose[:3], first_pose[3:])
    tp = torch.from_numpy(tp).unsqueeze(0).unsqueeze(0).cuda()
    ref_pose = get_pose_from_audio(tp, audio_feature, audio2pose_ckpt)
    torch.cuda.empty_cache()
    trans_seq = ref_pose[:, 3:]
    rot_seq = ref_pose[:, :3]



    audio_seq = audio_feature#[40:]
    ph_seq = phs["phone_list"]


    ph_frames = []
    audio_frames = []
    pose_frames = []
    name_len = frames

    pad = np.zeros((4, audio_seq.shape[1]), dtype=np.float32)


    print("GOT POSES, GENERATING FRAMES",time.time()-startTime)

    
    
    for rid in range(0, frames):
        ph = []
        audio = []
        pose = []
        for i in range(rid - opt.num_w, rid + opt.num_w + 1):
            if i < 0:
                rot = rot_seq[0]
                trans = trans_seq[0]
                ph.append(31)
                audio.append(pad)
            elif i >= name_len:
                ph.append(31)
                rot = rot_seq[name_len - 1]
                trans = trans_seq[name_len - 1]
                audio.append(pad)
            else:
                ph.append(ph_seq[i])
                rot = rot_seq[i]
                trans = trans_seq[i]
                audio.append(audio_seq[i * 4:i * 4 + 4])
            tmp_pose = np.zeros([256, 256])
            draw_annotation_box(tmp_pose, np.array(rot), np.array(trans))
            pose.append(tmp_pose)

        ph_frames.append(ph)
        audio_frames.append(audio)
        pose_frames.append(pose)

    
    audio_f = torch.from_numpy(np.array(audio_frames,dtype=np.float32)).unsqueeze(0)
    poses = torch.from_numpy(np.array(pose_frames, dtype=np.float32)).unsqueeze(0)
    ph_frames = torch.from_numpy(np.array(ph_frames)).unsqueeze(0)
   

    '''

    #this code works but is slow
    
    audio_pose_dataset = AudioPoseDataset(frames, audio_seq, ph_seq, rot_seq, trans_seq, opt)
    data_loader = DataLoader(audio_pose_dataset, batch_size=1, num_workers=4)

    ph_frames = []
    audio_frames = []
    pose_frames = []

    for ph, audio, pose in data_loader:
        ph_frames.append(ph.squeeze(0))
        audio_frames.append(audio.squeeze(0))
        pose_frames.append(pose.squeeze(0))

    audio_f=torch.stack(audio_frames).float().unsqueeze(0)
    poses=torch.stack(pose_frames).float().unsqueeze(0)
    ph_frames=torch.stack(ph_frames).unsqueeze(0)
    '''

    print("GENERATED FRAMES",time.time()-startTime)
    
    bs = audio_f.shape[1]
    predictions_gen = []

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()

    ph2kp = Audio2kpTransformer(opt).cuda()

    #keep kp_detector generator and ph2kp in global scope
    global _kp_detector
    global _generator
    global _ph2kp
    if _kp_detector is None:
        load_ckpt(generator_ckpt, kp_detector=kp_detector, generator=generator,ph2kp=ph2kp)
        _kp_detector = kp_detector
        _generator = generator
        _ph2kp = ph2kp
    else:
        kp_detector = _kp_detector
        generator = _generator
        ph2kp = _ph2kp

    print("\nLOADED CHECKPOINT",time.time()-startTime,"\n")


    ph2kp.eval()
    generator.eval()
    kp_detector.eval()

    kp_gen_source = kp_detector(img, True)

    with torch.no_grad():
        for frame_idx in range(0,bs,decimate):
            t = {}

            t["audio"] = audio_f[:, frame_idx].cuda()
            t["pose"] = poses[:, frame_idx].cuda()
            t["ph"] = ph_frames[:,frame_idx].cuda()
            t["id_img"] = img
            
            #kp_gen_source = kp_detector(img, True)

            gen_kp = ph2kp(t,kp_gen_source)
            if frame_idx == 0:
                drive_first = gen_kp

            norm = normalize_kp(kp_source=kp_gen_source, kp_driving=gen_kp, kp_driving_initial=drive_first)
            out_gen = generator(img, kp_source=kp_gen_source, kp_driving=norm)

            predictions_gen.append(
                (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))
            

    print("\nMADE PREDICTIONS",time.time()-startTime,"\n")


    log_dir = save_dir
    os.makedirs(os.path.join(log_dir, "temp"),exist_ok=True)

    f_name = os.path.basename(img_path)[:-4] + "_" + os.path.basename(audio_path)[:-4] + ".mp4"
    # kwargs = {'duration': 1. / 25.0}
    video_path = os.path.join(log_dir, "temp", f_name)
    print("save video to: ", video_path)
    imageio.mimsave(video_path, predictions_gen, fps=25.0/decimate)

    # audio_path = os.path.join(audio_dir, x['name'][0].replace(".mp4", ".wav"))
    save_video = os.path.join(log_dir, f_name)
    #cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, audio_path, save_video)
    #os.system(cmd)
    cmd = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-vcodec', 'copy', save_video]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    os.remove(video_path)

    print("saved video",time.time()-startTime)

    return save_video






if __name__ == '__main__':

    startTime=time.time()


    argparser = argparse.ArgumentParser()
    argparser.add_argument("--img_path", type=str, default=None, help="path of the input image ( .jpg ), preprocessed by image_preprocess.py")
    argparser.add_argument("--audio_path", type=str, default=None, help="path of the input audio ( .wav )")
    argparser.add_argument("--phoneme_path", type=str, default=None, help="path of the input phoneme. It should be note that the phoneme must be consistent with the input audio")
    argparser.add_argument("--save_dir", type=str, default="samples/results", help="path of the output video")
    argparser.add_argument("--phindex_location", type=str, default="phindex.json", help="path of phindex.json [default: phindex.json]")
    argparser.add_argument("--config_prefix", type=str, default="./", help="prefix of the config file [default: config]")
    args = argparser.parse_args()

    print("PARSE PHONEME FILE: ",time.time()-startTime)
    phoneme = parse_phoneme_file(args.phoneme_path,phindex_location=args.phindex_location)
    print("PROCESS IMAGE",time.time()-startTime)
    test_with_input_audio_and_image(args.img_path,args.audio_path,phoneme,config.GENERATOR_CKPT,config.AUDIO2POSE_CKPT,args.save_dir,config_prefix=args.config_prefix)

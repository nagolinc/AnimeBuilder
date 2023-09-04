from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def getllama(which_model="nous-llama2-7b"):

    if which_model == "nous-llama2-7b":
        model_name_or_path = "D:/img/llama/Nous-Hermes-Llama-2-7B-GPTQ"
        model_basename = "model"

        use_triton = False

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=False,
                device="cuda:0",
                use_triton=use_triton,
                quantize_config=None)
        

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

        return tokenizer,pipe


    if which_model=="nous":

        #load llama model
        quantized_model_dir = "D:\\img\\llama\\Nous-Hermes-13B-GPTQ"
        #quantized_model_dir = "D:\\img\\llama\\WizardLM-7B-uncensored-GPTQ"
        #quantized_model_dir = "D:\\img\\llama\\\mpt-7b-storywriter-4bit-128g"
        #quantized_model_dir = "D:\\img\\llama\\vicuna-7B-1.1-GPTQ-4bit-128g"



        llama_tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=False)

        def get_config(has_desc_act):
            return BaseQuantizeConfig(
                bits=4,  # quantize model to 4-bit
                group_size=128,  # it is recommended to set the value to 128
                #group_size=64,#falcon
                desc_act=has_desc_act
            )

        def get_model(model_base, triton, model_has_desc_act):
            if model_has_desc_act:
                model_suffix="latest.act-order"
            else:
                model_suffix="compat.no-act-order"
            model_suffix="no-act.order"#nous
            #model_suffix="compat.no-act-order"#wizard
            #model_suffix="no-act-order"#vicuna?
            model_basename=f"{model_base}.{model_suffix}"#wizard?+nous?
            #model_basename=f"{model_base}"#?
            return AutoGPTQForCausalLM.from_quantized(quantized_model_dir, 
                                                    use_safetensors=True, #nous
                                                    #use_safetensors=False, #vicuna
                                                    model_basename=model_basename, 
                                                    device="cuda:0", 
                                                    use_triton=triton, 
                                                    quantize_config=get_config(model_has_desc_act),
                                                    #trust_remote_code=True,#falcon/mpt
                                                    )
        
        llama_model = get_model("nous-hermes-13b-GPTQ-4bit-128g", triton=False, model_has_desc_act=False)
        #llama_model = get_model("WizardLM-7B-uncensored-GPTQ-4bit-128g", triton=False, model_has_desc_act=False)
        #llama_model = get_model("gptq_model-4bit-64g", triton=False, model_has_desc_act=False)
        #llama_model = get_model("model", triton=False, model_has_desc_act=False)
        #llama_model = get_model("vicuna-7B-1.1-GPTQ-4bit-128g", triton=False, model_has_desc_act=False)

        llm = pipeline(
            "text-generation",
            model=llama_model,
            tokenizer=llama_tokenizer,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            #trust_remote_code=True,#falcon/mpt
        )

        return llama_tokenizer,llm
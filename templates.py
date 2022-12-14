templates={
    "color":"""    
Output a list of colors

description:
red

description:
green

description:
blue

description:
{TEXT}""",
    "monster":"""
create a list of various fantasy monsters

description:
minotaur

description:
zombie

description:
gelatinous cube

description:
{TEXT}
    """,
    "character":"""
create a list of characters and their attributes

name:
Joe
gender:
male
hair color:
red
weapon:
tennis racket
description:
Joe is a fesity redhead who likes to play tennis

name:
Lily
gender:
female
hair color:
black
weapon:
syringe
description:
Lily is a cute girl with long black hair who studies biology

name:
{TEXT}
gender:
{TEXT}
hair color:
{color}
weapon:
{TEXT}
description:
{TEXT}""",
    "setting":"""
a list of movie settings

description:
a sunny beach

description:
a haunted mansion

description:
a trendy shop at the mall

description:
{TEXT}""",
    "screenplay":"""
write a short screenplay about two people

character1:
{character1}
character2:
{character2}
setting:
{setting}
transcript:
{character1.name}: {TEXT}
{character2.name}: {TEXT}
{character1.name}: {TEXT}
{character2.name}: {TEXT}
{character1.name}: {TEXT}
{character2.name}: {TEXT}



""",
    "fight":"""
write a short screenplay about two people fighting a monster

monster:
{monster}
character1:
{character1}
character2:
{character2}
setting:
{setting}
transcript:
{character1.name}: {TEXT}
{character2.name}: {TEXT}
[{character1.name} attacks the monster with their {character1.weapon}]
{character1.name}: {TEXT}
{character2.name}: {TEXT}
{character1.name}: {TEXT}
{character2.name}: {TEXT}



""",
        "adventure":"""
write a short screenplay about a hero overcoming great odds

character1:
{character1}
plot summary:
{character1.name} is the hero of an epic fantasy adventure.  Their goal is to {TEXT}.
In order to complete their goal, they will need to finish the following 3 steps
step 1:
{TEXT}
step 2:
{TEXT}
step 3:
{TEXT}



""",
            "story":"""
write a short screenplay

character1:
{character1}
character2:
{character2}
plot summary:
{character1.name} stars in this anime about {story synopsis:TEXT:}
subplot:
{subplot:TEXT:}
scene 1:
{scene 1:TEXT:}
scene 2:
{scene 2:TEXT:}
scene 3:
{scene 3:TEXT:}



""",
    "storyWithCharacters":"""
write a short screenplay

character1:
{character1}
character2:
{character2}
character3:
{character3}
character4:
{character4}
plot summary:
{character1.name} stars in this anime about {story synopsis:TEXT:}
subplot:
{subplot:TEXT:}
scene 1:
{character1.name} and {character2.name} {scene 1 text:TEXT:}
scene 2:
{character1.name} and {character3.name} {scene 2 text:TEXT:}
scene 3:
{character1.name} and {character4.name} {scene 3 text:TEXT:}



""",
    "advancePlot":"""
write a short screenplay

character1:
{character1}
character2:
{character2}
plot summary:
{character1.name} stars in this anime about {story synopsis:TEXT:}
subplot:
{subplot:TEXT:}
scene 1:
{scene 2 previous:TEXT:}
scene 2:
{scene 3 previous:TEXT:}
scene 3:
{character1.name} and {character2.name} {scene 3 text:TEXT:}



""",
    "plot overview":"""

character1:
{character1}
plot summary:
{character1.name} stars in this anime about {story synopsis:TEXT:}
part 1:
{part 1:TEXT:}
part 2:
{part 2:TEXT:}
part 3:
{part 3:TEXT:}
part 4:
{part 4:TEXT:}
part 5:
{part 5:TEXT:}



    
""",
    "descriptionToCharacter":"""
    
description:
Amy is a cute girl with brown hair who likes the beauty industry    
name:
Amy
gender:
female
weapon:
Scissors
hair color:
brown

description:
Dan is a male teen hacker with dark black hair who wears a turtleneck sweater
name:
Dan
gender:
male
weapon:
Keyboard
hair color:
dark black

description:
{description:TEXT:}
name:
{name:TEXT:}
gender:
{gender:TEXT:}
weapon:
{weapon:TEXT:}
hair color:
{hair color:TEXT:}
    
    
    
    """,
    "sceneToTranscript":"""



scene:
Amy and Jenna hang out after school
transcript:
Amy: School was so tough today!
Jenna: Don't worry, you'll get used to it!
Amy: I don't want to get used to it.  I want to go back to summer break
Jenna: Fair enough, but we have to go to school
Amy: Why? Isn't school just a prison for kids?
Jenna: Yeah, probably!


scene:
{scene:TEXT:}
setting:
{setting:TEXT:}
transcript:
{character1.name}: {line 1 text:TEXT:}
{character2.name}: {line 2 text:TEXT:}
{character1.name}: {line 3 text:TEXT:}
{character2.name}: {line 4 text:TEXT:}
{character1.name}: {line 5 text:TEXT:}
{character2.name}: {line 6 text:TEXT:}
    
    """,
    "sceneToCharacters":"""
    
    scene:
    Jane and Dave go to the park
    character1:
    Jane
    character2:
    Dave
    
    scene:
    Linda hangs out with her friend
    character1:
    Linda
    character2:
    Susan    
    
    scene:
    {scene:TEXT:}
    character1:
    {character1:TEXT:}
    character2:
    {character2:TEXT:}
    
    """,
    "fightScene":"""

write a short screenplay

character1:
{character1}
character2:
{character2}
plot summary:
{character1.name} stars in this anime about {story synopsis:TEXT:}
subplot:
{subplot:TEXT:}
scene 1:
{character1.name} and {character2.name} get ready to battle {monster}
scene 2:
{character1.name} and {character3.name} battle {monster}
scene 3:
{character1.name} and {character4.name} celebrate after defeating {monster}
    """,

}
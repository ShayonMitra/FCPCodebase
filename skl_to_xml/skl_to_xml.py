from pyparsing import (Literal, Word, alphanums, nums, Group, Optional, 
                       ZeroOrMore, OneOrMore, ParseException, alphas, 
                       Combine, Suppress, QuotedString, ParserElement,
                       restOfLine, delimitedList, Group, Suppress)
import json

TO_REFLECT = json.load(open("toReflect.json"))
REFLECTED_EFF = json.load(open("reflectedEff.json"))
EFF_TO_NAME = json.load(open("eff_to_name.json"))
NAME_TO_ACTUAL = json.load(open("name_to_actual.json"))
ACTUAL_TO_IDX = json.load(open("actual_to_idx.json"))

EFF_TO_IDX = {}
for eff in EFF_TO_NAME:
    EFF_TO_IDX[eff] = ACTUAL_TO_IDX[NAME_TO_ACTUAL[EFF_TO_NAME[eff]]]



# Define the Skill, KeyFrame, and other necessary classes
class Skill:

    def __init__(self):
        self.key_frames = []
        self.converted_key_frames = []
        self.skill_name = None
        self.auto_head = 1
    
    def append_key_frame(self, key_frame):
        self.key_frames.append(key_frame)
        
    def get_reflection(self, body_model=None):
        new_skill = Skill()
        for key_frame in self.key_frames:
            new_key_frame = KeyFrame()
            new_key_frame.set_wait_time(key_frame.wait_time)
            for macro in key_frame.macros:
                new_key_frame.append_macro(macro.get_reflection())
            new_skill.append_key_frame(new_key_frame)

        return new_skill  # Return a new skill as a placeholder
    
    def gen_converted_to_idx(self):
        self.converted_key_frames = []
        for key_frame in self.key_frames:
            converted_key_frame = key_frame.convert_to_idx()
            self.converted_key_frames.append(converted_key_frame)

    def convert_to_xml(self):
        self.gen_converted_to_idx()
        xml = """<?xml version="1.0" encoding="UTF-8"?>"""
        xml += "\n\n"
        xml += f"<behavior description=\"{self.skill_name}\" auto_head=\"{self.auto_head}\">"
        for key_frame in self.converted_key_frames:
            xml += "\n\t"
            xml += f"<slot delta=\"{key_frame.wait_time}\">"
            for macro in key_frame.macros:
                if isinstance(macro, SetTar):
                    for effector, value in macro.effector_pairs:
                        xml += f"\n\t\t<move id=\"{effector}\" angle=\"{value}\" />"
            xml += "\n\t</slot>"
        
        xml += "\n</behavior>"
        return xml



class KeyFrame:
    def __init__(self):
        self.macros = []
        self.wait_time = 0
        
    def append_macro(self, macro):
        self.macros.append(macro)
    
    def set_wait_time(self, wait_time):
        self.wait_time = wait_time
        
    def convert_to_idx(self):
        converted_macros = []
        converted_key_frame = KeyFrame()
        converted_key_frame.wait_time = self.wait_time
        for macro in self.macros:
            converted_macros.append(macro.convert_to_idx())
        converted_key_frame.macros = converted_macros
        return converted_key_frame


class Macro:
    pass

class Reset(Macro):
    def __init__(self, joint_indices):
        self.joint_indices = joint_indices

    def get_reflection(self):
        return Reset(self.joint_indices)

class SetTar(Macro):
    def __init__(self, effector_pairs):
        self.effector_pairs = effector_pairs
    
    def get_reflection(self):
        reflected_effector_pairs = []
        for effector, value in self.effector_pairs:
            reflected_effector_pairs.append((REFLECTED_EFF[effector], -value if TO_REFLECT[effector] else value))
        return SetTar(reflected_effector_pairs)
    
    def convert_to_idx(self):
        converted_set_tar = SetTar([])
        for effector, value in self.effector_pairs:
            converted_set_tar.effector_pairs.append((EFF_TO_IDX[effector], value))
        return converted_set_tar


# Define the parser
ParserElement.enablePackrat()  # For efficiency

# Define tokens
identifier = Word(alphas + '_', alphanums + '_')
number = Combine(Optional('-') + Word(nums) + Optional('.' + Word(nums)))
value = number.setParseAction(lambda t: float(t[0]))

# Define the grammar
start_skill = Literal("STARTSKILL").suppress()
end_skill = Literal("ENDSKILL").suppress()
start_state = Literal("STARTSTATE").suppress()
end_state = Literal("ENDSTATE").suppress()
start_curve = Literal("STARTCURVE").suppress()
end_curve = Literal("ENDCURVE").suppress()
reset = Literal("reset").suppress()
end = Literal("end").suppress()
wait = Literal("wait").suppress()
set_tar = Literal("settar").suppress()
set_foot = Literal("setfoot").suppress()
stabilize = Literal("stabilize").suppress()

# Define callback functions
def start_skill_cb(skillname, parser):
    parser.current_skill_type = skillname
    parser.skills[skillname] = Skill()
    parser.skills[skillname].skill_name = skillname

def reflect_skill_cb(source, target, parser):
    if source in parser.skills:
        parser.skills[target] = parser.skills[source].get_reflection()
        parser.skills[target].skill_name = target

def start_key_frame_cb(parser):
    parser.current_key_frame = KeyFrame()
    if parser.current_skill_type in parser.skills:
        parser.skills[parser.current_skill_type].append_key_frame(parser.current_key_frame)

def reset_cb(joints, key_frame):
    joint_indices = [parser.enum_parser.get_enum_from_string(joint) for joint in joints]
    reset_macro = Reset(joint_indices)
    key_frame.append_macro(reset_macro)

def wait_cb(wait_time, key_frame):
    key_frame.set_wait_time(wait_time)

def set_tar_cb(effector_pairs, key_frame):
    settar_macro = SetTar(effector_pairs)
    key_frame.append_macro(settar_macro)


# Define the grammar rules
skillname = identifier.setResultsName('skillname')
ref_skill_source = identifier.setResultsName('ref_skill_source')
ref_skill_target = identifier.setResultsName('ref_skill_target')
joints = OneOrMore(identifier).setResultsName('joints')
effectors = OneOrMore(identifier).setResultsName('effectors')
eff_values = OneOrMore(value).setResultsName('eff_values')
positions = OneOrMore(value).setResultsName('positions')
foot = identifier.setResultsName('foot')
wait_time = value.setResultsName('wait_time')

# Define the pattern for effector-value pairs
effector_value_pair = Group(identifier + value)
effector_value_pairs = OneOrMore(effector_value_pair).setResultsName('effector_pairs')

reset_macro = (reset + OneOrMore(identifier) + end).setParseAction(lambda t: reset_cb(t[0], parser.current_key_frame))

wait_macro = (wait + wait_time + end).setParseAction(lambda t: wait_cb(t.wait_time, parser.current_key_frame))
set_tar_macro = (set_tar + effector_value_pairs + end).setParseAction(lambda t: set_tar_cb(t.effector_pairs, parser.current_key_frame))

start_state_stmt = (start_state).setParseAction(lambda: start_key_frame_cb(parser))
state = (start_state + OneOrMore(reset | wait_macro | set_tar_macro) +
         end_state)

# Define the grammar rules
start_skill_stmt = (start_skill + skillname).setParseAction(lambda t: start_skill_cb(t.skillname, parser))
skill = (start_skill_stmt +
         OneOrMore(state) +
         end_skill)

reflect_skill = (Literal("REFLECTSKILL") + ref_skill_source + ref_skill_target).setParseAction(lambda t: reflect_skill_cb(t.ref_skill_source, t.ref_skill_target, parser))

# Main parsing rule
top = (OneOrMore(skill | reflect_skill))

def parse(text, parser):
    try:
        parsed = top.parseString(text, parseAll=True)
        print("Parsing succeeded")
    except ParseException as pe:
        print(f"Parsing failed: {pe}")

# Example usage
skills = {}
body_model = None
parser = type('Parser', (), {'skills': skills, 'body_model': body_model, 'current_skill_type': None, 'current_key_frame': None, 'enum_parser': None})

def convert_to_xml(skill: Skill):
    pass

if __name__ == "__main__":
    filename = "kick_long_15.skl"
    with open (filename, "r") as file:
        text = file.read()
        parse(text, parser)

    for skill_name, skill in parser.skills.items():
        print("Skill:", skill_name)
        # for key_frame in skill.key_frames:
        #     print("Wait time:", key_frame.wait_time)
            # for macro in key_frame.macros:
            #     if isinstance(macro, SetTar):
            #         print(macro.effector_pairs)
            #     elif isinstance(macro, Reset):
            #         print(macro.joint_indices)
            #     else:
            #         print("Unknown macro type")
        xml = skill.convert_to_xml()
        with open(f"{skill_name}_converted.xml", "w") as file:
            file.write(xml)

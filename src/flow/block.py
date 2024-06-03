

class Block(object):
    
    def __init__(self, des):
        '''
        des format: <name>:::<type>:::<instruction>:::<branch>
        <type>: precess or decision or terminal
        <branch>: <key1>::<value1>::<key2>::<value2>::...
        Example: Step 1:::decision:::Check whether every models in the generated to-do list is in the provided models:::Yes::step 5::No::step 3
        '''
        structured_info = self.parser(des)
        self.name = structured_info['name'].lower()
        self.instruction = structured_info['instruction']
        self.type = structured_info['type'].lower()
        self.branch = structured_info['branch']
    
    def parser(self, des):
        '''
        Parse the block description into structured information.
        
        Args:
            des: the description of the block.
            
        Returns:
            structured_info: ('Dict[str,Any]'):
                name: str,
                type: str,
                instruction: str,
                branch: List[str]
        '''
        structured_info = dict()
        info = [component.strip() for component in des.split(":::")]
        structured_info['name'] = info[0]
        
        # Check whether type info is correct
        if info[1].lower() not in ['process', 'decision', 'terminal']:
            raise Exception("The type is not in [process, decision, terminal]")
        
        structured_info['type'] = info[1]
        structured_info['instruction'] = info[2]
        structured_info['branch'] = dict()
        branch_info = [branch.strip() for branch in info[3].split("::")]
        num = len(branch_info) // 2
        for i in range(num):
            structured_info['branch'][branch_info[2 * i].lower()] = branch_info[2 * i + 1].lower()
            
        return structured_info
    
    def __str__(self):
        
        branch_str = ''
        for key, value in self.branch.items():
            if isinstance(value, str):
                branch_str += f'if {key}, then go to {value}. '
            else:
                branch_str += f'if {key}, then go to {value.name}. '
        
        return f'{self.name}, the type is {self.type}, the instruction is {self.instruction} {branch_str}\n'

    def get_instruction(self):
        return self.instruction
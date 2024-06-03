from flow.block import Block
from utils import flow_utils

class Flow(object):
    
    def __init__(self, flow_file):
        '''
        Construct the flow based on the descriptions from flow_file
        
        Args:
            flow_file (str): the path pointed to the text file containing the flow instructions.
        '''
        
        flow_instruction = flow_utils.ReadLineFromFile(flow_file)
        
        self.block_dict = dict()
        self.header = None
        
        for row in flow_instruction:
            step_block = Block(row)
            self.block_dict[step_block.name] = step_block
            
            if self.header is None:
                self.header = step_block
                
        self.connect_blocks()
        
        
    def connect_blocks(self):
        '''
        Connect blocks
        '''
        for key, value in self.block_dict.items():
            try:
                for branch_condition, branch_block in value.branch.items():
                    value.branch[branch_condition] = self.block_dict[branch_block]
            except:
                raise Exception("Error when connecting blocks in flow")
        
    def __str__(self):
        '''
        Return the flow information as string.
        '''
        flow_str = ''
        for key, value in self.block_dict.items():
            flow_str += value.__str__()
        return flow_str
        
if __name__ == '__main__':
    flow = Flow('../OpenAGI_Flow.txt')
    print(flow.__str__())
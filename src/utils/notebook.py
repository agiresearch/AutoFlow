from pandas import DataFrame

class Notebook:
    def __init__(self) -> None:
        self.data = dict()

    def write(self, key:str, input_data: DataFrame, short_description: str):
        if key in self.data:
            self.update(key, input_data, short_description)
        else:
            self.data[key] = {"Short Description": short_description, "Content":input_data}
            return f"The information ```{short_description}``` has been recorded in Notebook, and its key is {key}. The content is \n{input_data}"
    
    def update(self, key:str, input_data: DataFrame, short_decription: str):
        self.data[key]["Content"] = input_data
        self.data[key]["Short Description"]  = short_decription

        return f"The information has been updated in Notebook."
    
    def list(self):
        results = []
        for key, unit in self.data.items():
            results.append({"key":key, "Short Description":unit['Short Description']})
        
        return results

    def list_all(self):
        results = []
        for key, unit in self.data.items():
            if type(unit['Content']) == DataFrame:
                results.append({"key":key, "Short Description":unit['Short Description'], "Content":unit['Content'].to_string(index=False)})
            else:
                results.append({"key":key, "Short Description":unit['Short Description'], "Content":unit['Content']})
        
        return results
    
    def list_all_str(self):
        results = []
        for key, unit in self.data.items():
            if type(unit['Content']) == DataFrame:
                results.append(f"key: {key}, Short Description: {unit['Short Description']}, Content: {unit['Content'].to_string(index=False)}")
            else:
                results.append(f"key: {key}, Short Description: {unit['Short Description']}, Content: {unit['Content']}")
        
        return results
    
    def list_keys(self):
        return list(self.data.keys())
    
    def to_str(self, key:str):
        info = self.data[key]
        if type(info['Content']) == DataFrame:
            return f"{key}: {info['Short Description']}\n{info['Content'].to_string(index=False)}\n\n"
        else:
            return f"{key}: {info['Short Description']}\n{info['Content']}\n"

    def read(self, key:str):
        return self.data[key]
    
    def reset(self):
        self.data = dict()
    
    

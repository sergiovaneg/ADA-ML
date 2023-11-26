import json
import pandas as pd
from datetime import datetime

date_format = "%Y%m%d%H%M%S"
model_list = []
recover_file = "./results_bak.txt"
with open(recover_file, "r", encoding="UTF8") as file_reader:
  for line in file_reader:
    edited_line = line.replace("Model: ", "").replace("'", '"')
    current_dict = json.loads(edited_line)
    model_list.append(current_dict)

model_df = pd.DataFrame.from_dict(model_list)
model_df.to_excel(f"./results_{datetime.now().strftime(date_format)}.xlsx")
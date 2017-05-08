import pandas as pd

tmp = []
for i in range(2345796):
    tmp.append({'test_id':i,'is_duplicate':0.5})

df = pd.DataFrame(tmp)
df.to_csv('all_zero_five.csv', ',', columns=['test_id', 'is_duplicate'], index=False)
my_json =  {"name":"python","address":{"province":"山西省","city":["吕梁市","晋中市","太原市"]}}

# 获取山西省
province = my_json["address"]["province"]
print(province)

# 获取吕梁市
city = my_json["address"]["city"][0]
print(city)
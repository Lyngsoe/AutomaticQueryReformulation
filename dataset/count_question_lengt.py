from trec_car.read_data import *
from car import CAR

car = CAR(debug=False)

queries = car.get_query_parts()

q_len = 0

for query in queries:
    q_len += len(query)

print("total query character length",q_len)
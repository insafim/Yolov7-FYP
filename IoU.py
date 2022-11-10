from shapely.geometry import Polygon
import itertools

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


Dict = {}
Dict["1"] = [[511, 41], [577, 41], [577, 76], [511, 76]]
Dict["2"] = [[544, 59], [610, 59], [610, 94], [544, 94]]
Dict["3"] = [[530, 50], [610, 50], [610, 84], [530, 84]]

print(Dict)
#box_1 = [[511, 41], [577, 41], [577, 76], [511, 76]]
#box_2 = [[544, 59], [610, 59], [610, 94], [544, 94]]

result_list = list(map(dict, itertools.combinations(Dict.items(), 2)))

#print(result_list)
#print(result_list[0])

List = []
xmin,xmax,ymin,ymax = 1,1,1,1
for i in range(5):
    xmin += i + 5
    xmax += i + 20
    ymin += i + 3
    ymax += i + 30
    ele = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    List.append(ele)

#print(List)
result_list2 = list(itertools.combinations(List, 2))
print(result_list2[1][0])
print(result_list2[1][1])

print(calculate_iou(result_list2[1][0], result_list2[1][1]))

for i in range(len(result_list2)):
    print(calculate_iou(result_list2[i][0], result_list2[i][1]))
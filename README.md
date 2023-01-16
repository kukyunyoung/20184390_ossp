# 20184390_ossp

## 1월13일
![image](https://user-images.githubusercontent.com/71830573/212336662-94aa974c-6095-404b-ab41-2b6af3a4ee1b.png)


이미지를 구성하는 색상의 팔레트를 나열하는 사진.
랜드마크 좌표를 이용하여 눈의 일부분을 범위로 지정하여 오늘 추가한 코드를 이용해 눈동자의 색상을 판별할 것임.


## 1월16일
![image](https://user-images.githubusercontent.com/71830573/212673327-e83a6ffe-a9fc-4d0b-a68b-354b106e03e2.png)
![image](https://user-images.githubusercontent.com/71830573/212673342-75842ce1-2098-4140-8ce5-4ff3e8f54b89.png)
![image](https://user-images.githubusercontent.com/71830573/212673359-799e15ee-3076-467a-81d8-e2a98935f421.png)


2,3번째 사진은 첫번째사진의 왼쪽눈과 오른쪽눈을 크롭하여 팔레트의 색상을 비교한것

눈동자를 가리키는 랜드마크의 x,y좌표를 이용하여 적당한 범위를 지정해준 뒤 그 범위안의 이미지에 대해 색상추출을 진행함

```python
lefteye_crop_img = ori_img[ori_lmks[0][1]-20:ori_lmks[0][1]+40, ori_lmks[0][0]-20:ori_lmks[0][0]+20] # 왼쪽눈 크롭
righteye_crop_img = ori_img[ori_lmks[1][1]-20:ori_lmks[1][1]+40, ori_lmks[1][0]-20:ori_lmks[1][0]+30] # 오른쪽눈 크롭
```

크롭한 고양이의 눈동자 사진에서 라이브러리 함수를 사용하여 눈동자의 색상군 분포를 추출함

```python
# 고양이 사진을 가져와서 랜드마크점을 이용하여 눈주위 식별 및 색판별
lefteye_img = lefteye_crop_img.copy() # 원래이미지 가져오기
lefteye_img = cv2.cvtColor(lefteye_img, cv2.COLOR_BGR2RGB) # bgr 에서 rgb로 바꿔줌
dim = (500,300)
lefteye_img = cv2.resize(lefteye_img, dim, interpolation=cv2.INTER_AREA)

clt = KMeans(n_clusters=5) # n_cluster 조정해서 추출할 색상군의 수를 정할 수 있음 (너무많은 색상군이 검출되면 응용할 것)
clt.fit(lefteye_img.reshape(-1,3))
print("clt.labels_ \n")
clt.labels_
print("clt.cluster_centers_ \n")
clt.cluster_centers_

righteye_img = righteye_crop_img.copy() # 원래이미지 가져오기
righteye_img = cv2.cvtColor(righteye_img, cv2.COLOR_BGR2RGB) # bgr 에서 rgb로 바꿔줌
dim = (500,300)
righteye_img = cv2.resize(righteye_img, dim, interpolation=cv2.INTER_AREA)

clt = KMeans(n_clusters=5) # n_cluster 조정해서 추출할 색상군의 수를 정할 수 있음 (너무많은 색상군이 검출되면 응용할 것)
clt.fit(righteye_img.reshape(-1,3))
print("clt.labels_ \n")
clt.labels_
print("clt.cluster_centers_ \n")
clt.cluster_centers_
```

위에서 추출한 눈동자의 색상군 분포를 팔레트형식으로 출력해주기위한 함수를 정의함

```python
def palette_perc(k_cluster):
        width=300
        palette = np.zeros((50,width, 3), np.uint8)

        n_pixels = len(k_cluster.labels_)
        counter = Counter(k_cluster.labels_)
        perc = {}
        for i in counter:
            perc[i] = np.round(counter[i]/n_pixels, 2)
        perc = dict(sorted(perc.items()))

        print(perc)
        print(k_cluster.cluster_centers_)

        step = 0

        for idx, centers in enumerate(k_cluster.cluster_centers_):
            palette[:,step:int(step+perc[idx]*width+1),:] = centers
            step += int(perc[idx]*width+1)
        
        return palette
```

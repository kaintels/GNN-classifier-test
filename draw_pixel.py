import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
from sklearn.neighbors import NearestNeighbors

# 1. 이미지 불러오기 및 슈퍼픽셀 생성
def generate_superpixels(image, n_segments=100):
    """
    이미지에서 SLIC 알고리즘을 사용하여 슈퍼픽셀을 생성.
    n_segments: 생성할 슈퍼픽셀의 수
    """
    segments = slic(image, n_segments=n_segments, compactness=50, sigma=1, start_label=1)
    return segments

# 2. 슈퍼픽셀의 중심 좌표 및 색상 정보를 추출
def get_superpixel_features(image, segments):
    """
    각 슈퍼픽셀의 중심 좌표 및 평균 색상 값을 추출.
    """
    superpixel_centers = []
    superpixel_colors = []

    for segment_label in np.unique(segments):
        # 슈퍼픽셀의 모든 픽셀 좌표
        mask = (segments == segment_label)
        coords = np.column_stack(np.where(mask))

        # 중심 좌표 계산
        center = np.mean(coords, axis=0)
        superpixel_centers.append(center)

        # 슈퍼픽셀의 평균 색상 계산
        mean_color = np.mean(image[mask], axis=0)
        superpixel_colors.append(mean_color)
    
    return np.array(superpixel_centers), np.array(superpixel_colors)

# 3. K-근접 이웃을 이용해 슈퍼픽셀 간 연결 생성
def generate_rag(superpixel_centers, k=5):
    """
    K-근접 이웃을 사용하여 슈퍼픽셀 간의 연결을 생성.
    k: 각 슈퍼픽셀이 연결할 이웃의 개수
    """
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(superpixel_centers)
    distances, neighbors = knn.kneighbors(superpixel_centers)
    return neighbors

# 3. 좌표와 색상 정보를 결합한 특징 벡터 생성
def create_feature_vector(superpixel_centers, superpixel_colors):
    # 좌표와 색상 정보를 결합하여 특징 벡터 생성
    feature_vector = np.hstack((superpixel_centers, superpixel_colors))
    return feature_vector

# 4. 결과 시각화
def visualize_superpixels(image, segments):
    """
    원본 이미지 위에 슈퍼픽셀 경계를 표시하여 시각화.
    """
    segmented_image = label2rgb(segments, image, kind='avg')
    plt.imshow(segmented_image)
    # plt.title("Superpixel Segmentation")
    # plt.axis('off')
    plt.show()

def visualize_graph(superpixel_centers, neighbors, superpixel_colors):
    plt.figure(figsize=(10, 10))
    for i, neighbor_indices in enumerate(neighbors):
        for neighbor in neighbor_indices:
            # 노드 간 연결
            plt.plot([superpixel_centers[i][1], superpixel_centers[neighbor][1]],
                     [superpixel_centers[i][0], superpixel_centers[neighbor][0]], 'k-', alpha=0.3)

    # 각 노드 시각화 (슈퍼픽셀의 평균 색상을 반영하여 노드 색상 표현)
    plt.scatter(superpixel_centers[:, 1], superpixel_centers[:, 0], c=superpixel_colors / 255, s=100)

    plt.gca().invert_yaxis()
    # plt.title("Superpixel Graph with K-nearest Neighbors")
    plt.show()

# 5. 메인 함수
if __name__ == "__main__":
    # 이미지 불러오기
    image = cv2.imread("./pet-5137124_640.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 슈퍼픽셀 생성
    segments = generate_superpixels(image, n_segments=300)

    # 슈퍼픽셀 특징 추출 (중심 좌표, 색상)
    superpixel_centers, superpixel_colors = get_superpixel_features(image, segments)

    feature_vector = create_feature_vector(superpixel_centers, superpixel_colors)
    # K-근접 이웃을 이용한 슈퍼픽셀 간 연결 생성
    neighbors = generate_rag(feature_vector, k=5)

    # 결과 출력 (이웃 정보 출력)
    for i, neighbor_indices in enumerate(neighbors):
        print(f"슈퍼픽셀 {i}의 이웃: {neighbor_indices}")

    # 슈퍼픽셀 시각화
    visualize_superpixels(image, segments)
    visualize_graph(superpixel_centers, neighbors, superpixel_colors)
    plt.imshow(image)
    plt.show()

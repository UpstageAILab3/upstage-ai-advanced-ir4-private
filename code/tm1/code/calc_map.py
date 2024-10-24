import os
import pandas as pd
import argparse

def calc_map(gt, pred):    
    """
    Ground truth(gt)와 예측값(pred)을 기반으로 변형된 MAP 점수를 계산합니다.
    """
    sum_average_precision = 0

    for j in pred:  # 각 질의에 대해 반복
        eval_id = j["eval_id"]

        # Ground truth에 해당 eval_id가 있는지 확인
        if eval_id in gt and gt[eval_id]:  # Ground truth 문서가 있는 경우
            hit_count = 0  # 적중한 문서 수
            sum_precision = 0  # 정밀도의 합

            # top 3개의 예측 문서에 대해 반복하며 적중 여부 확인
            for i, docid in enumerate(j["topk"][:3]):
                if docid in gt[eval_id]:  # 예측한 문서가 정답인 경우
                    hit_count += 1
                    sum_precision += hit_count / (i + 1)

            # 평균 정밀도 계산
            average_precision = sum_precision / hit_count if hit_count > 0 else 0
        
        else:  # 과학 상식 질문이 아닌 경우
            average_precision = 0 if j["topk"] else 1  # 검색 결과가 없으면 1점, 있으면 0점
        
        sum_average_precision += average_precision

    # 모든 질의에 대한 평균 정밀도(MAP)를 계산
    return sum_average_precision / len(pred)

def load_ground_truth(filepath):
    """
    CSV 파일에서 Ground Truth 데이터를 불러와서 딕셔너리로 변환합니다.
    """
    gt_df = pd.read_csv(filepath)
    gt_dict = {}
    
    for _, row in gt_df.iterrows():
        gt_dict[row['eval_id']] = row['doc_ids'].split(',')  # 콤마로 구분된 문서들을 리스트로 변환
    
    return gt_dict

def load_predictions(filepath):
    """
    CSV 파일에서 예측 결과 데이터를 불러와서 리스트 형태로 변환합니다.
    """
    pred_df = pd.read_csv(filepath)
    pred_list = []

    for _, row in pred_df.iterrows():
        topk_list = row['topk'].split(',') if pd.notna(row['topk']) else []  # topk가 비어있을 수 있으니 체크
        pred_list.append({
            "eval_id": row['eval_id'],
            "topk": topk_list
        })

    return pred_list

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate MAP score for the competition.")
    parser.add_argument('--gt', type=str, required=True, help="Path to the ground truth CSV file.")
    parser.add_argument('--pred', type=str, required=True, help="Path to the prediction CSV file.")
    args = parser.parse_args()

    # 파일에서 데이터 불러오기
    gt = load_ground_truth(args.gt)
    pred = load_predictions(args.pred)

    # MAP 계산
    map_score = calc_map(gt, pred)
    print(f"MAP Score: {map_score:.4f}")

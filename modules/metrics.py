from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


# def compute_scores(gts, res):
#     """
#     Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

#     :param gts: Dictionary with the image ids and their gold captions,
#     :param res: Dictionary with the image ids ant their generated captions
#     :print: Evaluation score (the mean of the scores of all the instances) for each measure
#     """

#     # Set up scorers
#     scorers = [
#         (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
#         # (Meteor(), "METEOR"),
#         (Rouge(), "ROUGE_L"),
#         (Cider(),"CIDEr")
#     ]
#     eval_res = {}
#     # Compute score for each metric
#     for scorer, method in scorers:
#         try:
#             score, scores = scorer.compute_score(gts, res, verbose=0)
#         except TypeError:
#             score, scores = scorer.compute_score(gts, res)
#         if type(method) == list:
#             for sc, m in zip(score, method):
#                 eval_res[m] = sc
#         else:
#             eval_res[method] = score
#     return eval_res

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions
    :param res: Dictionary with the image ids and their generated captions
    :return: Dictionary of evaluation scores for each metric
    """

    assert gts.keys() == res.keys(), "Mismatch in image IDs between gts and res"

    # Safeguard: skip scoring if empty or invalid predictions
    if not gts or not res:
        print("[Warning] Empty gts or res input.")
        return {
            "BLEU_1": 0.0, "BLEU_2": 0.0, "BLEU_3": 0.0, "BLEU_4": 0.0,
            "ROUGE_L": 0.0, "CIDEr": 0.0
        }

    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    eval_res = {}

    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except Exception as e:
            print(f"[Warning] {scorer.__class__.__name__} scoring failed: {e}")
            score = [0.0] * len(method) if isinstance(method, list) else 0.0

        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score

    return eval_res

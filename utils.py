from fkassim.FastKassim import FastKassim
from nltk.tree import Tree
from typing import List, Union, Dict, Any

def f1_score(pred:List[str], ref:List[str], return_error:bool=False) -> Dict[str, Any]:
    _ref = ref[:]
    individual_result = {}  
    
    tp, len_pred, len_ref = 0, len(pred), len(_ref)
    incorrect_pred = []
    for relation in pred:
        if relation in _ref:
            _ref.remove(relation)
            tp += 1
        elif return_error:
            incorrect_pred.append(relation)
            
    if return_error:
        incorrect_ref = _ref
            
    precision, recall, f1_score = 0, 0, 0   
    if len_pred != 0:
        precision = tp / len_pred
    if len_ref != 0:
        recall = tp / len_ref

    if not precision and not recall:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    individual_result['tp'] = tp
    individual_result['len_pred'] = len_pred
    individual_result['len_ref'] = len_ref
    individual_result['precision'] = precision
    individual_result['recall'] = recall
    individual_result['f1_score'] = f1_score

    if return_error:
        individual_result['incorrect_pred'] = incorrect_pred
        individual_result['incorrect_ref'] = incorrect_ref
        
    return individual_result


def micro_f1_score(
    total_tp: int, 
    total_len_pred: int, 
    total_len_ref: int, 
    )->Dict[str, Any]:
    overall_result = {}
    
    precision = total_tp / total_len_pred if total_len_pred > 0 else 0
    recall = total_tp / total_len_ref if total_len_ref > 0 else 0

    if not precision and not recall:
        micro_f1_score = 0
    else:
        micro_f1_score = 2 * precision * recall / (precision + recall)

    overall_result["total_precision"] = precision
    overall_result["total_recall"] = recall
    overall_result["micro_f1_score"] = micro_f1_score

    return overall_result


class CustomFastKassim(FastKassim):
    def parse_document_with_index(self, doc_with_index:List[List[Union[int, str]]]=None, tokenizer=None, parser=None):
        index = doc_with_index[0]
        doc = doc_with_index[1]
        if tokenizer is None:
            tokenizer = self.sent_detector.tokenize
        if parser is None:
            parser = self.parser.raw_parse_sents
        
        parsed_sent = []
        doc_sents = tokenizer(doc.strip())
        try:
            doc_parsed = parser((doc_sents))
        except Exception as e:
            print(e)
            return []
        doc_parsed = list(doc_parsed)
        for i in range(len(doc_parsed)):
            doc_parsed_i = list(doc_parsed[i])[0]
            parsed_sent_i = Tree.convert(doc_parsed_i)
            parsed_sent.append(parsed_sent_i)
        doc_with_index = [index, parsed_sent]
        return doc_with_index
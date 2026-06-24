python hico_pipe/Initial_HOI_Identification.py
python hico_pipe/Initial_HOI_Identification.py
python hico_pipe/Initial_HOI_Identification.py
python hico_pipe/Initial_HOI_Identification.py


python hico_pipe/structured_answer.py --stage 1
python hico_pipe/filter_object.py --stage 1

python hico_pipe/HOI_Remining.py
python hico_pipe/HOI_Remining.py
python hico_pipe/HOI_Remining.py
python hico_pipe/HOI_Remining.py


python hico_pipe/structured_answer.py --stage 2
python hico_pipe/filter_object.py --stage 2


python hico_pipe/combine_same_HOI.py
python hico_pipe/object_logits_update.py


python hico_pipe/Action_Reassignment.py
python hico_pipe/Action_Reassignment.py
python hico_pipe/Action_Reassignment.py
python hico_pipe/Action_Reassignment.py


python hico_pipe/extract_ic_logits.py


# python hico_pipe/outbox.py
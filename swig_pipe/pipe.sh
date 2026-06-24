python swig_pipe/Initial_HOI_Identification.py
python swig_pipe/Initial_HOI_Identification.py
python swig_pipe/Initial_HOI_Identification.py
python swig_pipe/Initial_HOI_Identification.py


python swig_pipe/structured_answer.py --stage 1
python swig_pipe/filter_object.py --stage 1

python swig_pipe/HOI_Remining.py
python swig_pipe/HOI_Remining.py
python swig_pipe/HOI_Remining.py
python swig_pipe/HOI_Remining.py


python swig_pipe/structured_answer.py --stage 2
python swig_pipe/filter_object.py --stage 2


python swig_pipe/combine_same_HOI.py
python swig_pipe/object_logits_update.py


python swig_pipe/Action_Reassignment.py
python swig_pipe/Action_Reassignment.py
python swig_pipe/Action_Reassignment.py
python swig_pipe/Action_Reassignment.py


python swig_pipe/extract_ic_logits.py


python swig_pipe/outbox.py
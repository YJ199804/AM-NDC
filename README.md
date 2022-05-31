# AM-NDC
## Requirements
Please install pakeages by 
```javascript 
pip install -r requirements.txt
```
## Usage Example
Cora
```javascript 
python main.py --dataset Cora --epochs 500 --lr 5e-2 --weight_decay 5e-4 --dropout 0.3 --hidden_dim 128 --alpha 1.5 --tau 2 --k 2
```
CiteSeer
```javascript 
python main.py --dataset CiteSeer --epochs 600 --lr 3e-2 --weight_decay 1e-3 --dropout 0.5 --hidden_dim 200 --alpha 0.5 --tau 0.5 --k 3
```
PubMed
```javascript 
python main.py --dataset PubMed --epochs 600 --lr 1e-2 --weight_decay 2e-4 --dropout 0.2 --hidden_dim 200 --alpha 1 --tau 1 --k 3
```
Amazon Photo
```javascript 
python main.py --dataset photo --epochs 500 --lr 5e-3 --weight_decay 1e-3 --dropout 0.3 --hidden_dim 200 --alpha 1 --tau 4 --k 4
```
Amazon Computers
```javascript 
python main.py --dataset computers --epochs 500 --lr 1.5e-3 --weight_decay 1e-3 --dropout 0.4 --hidden_dim 500 --alpha 1.5 --tau 4 --k 2
```

## Results
model	|Cora	|CiteSeer	|PubMed|Amazon Photo		|Amazon Computers
------ | -----  |----------- |---|--- | -----  |
AM-NDC|	84.1% |	74.8%|	82.1%|93.3%|	83.4% |

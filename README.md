# local_global_survival

the main.py is the main file to conduct experiment.

in main.py the "Cindex_type" and "measure" can be used to select the proposed loss combinations i.e.:

Uno+NMAE:
```
model  = CombinedLossSurvModel(net, sigma=0.01, Cindex_type="ipcw", 
                    event_train= event_all,alpha=0.5,
                    time_train = time_all, measure='mae',dtype=dtype).to(device)
```
Uno+NMSE:
```
model  = CombinedLossSurvModel(net, sigma=0.01, Cindex_type="ipcw", 
                    event_train= event_all,alpha=0.5,
                    time_train = time_all, measure='mse',dtype=dtype).to(device)
```
Harrell+NMAE:
```
model  = CombinedLossSurvModel(net, sigma=0.01, Cindex_type="Harrell", 
                    event_train= event_all,alpha=0.5,
                    time_train = time_all, measure='mae',dtype=dtype).to(device)
```
Harrell+NMSE:
```
model  = CombinedLossSurvModel(net, sigma=0.01, Cindex_type="Harrell", 
                    event_train= event_all,alpha=0.5,
                    time_train = time_all, measure='mse',dtype=dtype).to(device)
```

# FlowInpaint
An image inpainting algorithm for serial sectioning images guided by optical flow from neighbouring images


#### step1:Initial reference module
```python

python get_interpolation.py
```
#### step2:Refined reference module
```python

flow_train.py
```
#### step3:Guided inpainting module
```python

fuse_train.py
```

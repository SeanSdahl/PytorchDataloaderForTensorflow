# PytorchDataloaderForTensorflow
## Use
As the PyTorch Dataloader has some transforms for input images that can not be done with <code>tf.keras</code> transforms easily it is useful to be able to load image data with a PyTorch dataloader even for fitting a <code>tf.keras</code> model. Therefore a class is implemented that uses a PyTorch dataloader object (doing the transformation on the data) which can be fed into the <code>tf.keras.model.fit_generator</code> function, to provide the training data for the <code>tf.keras</code> model.
## Setup
The python files were created for python version 3.7, although it might also work for past or future versions.
To use this class, some python modules need to be installed first. Using <code>pip</code> the packages can be installed by either typing 
<code>pip install -r requirements.txt</code>
in terminal, if the requirements.txt file exists in the current working directory or by typing
<code>pip install tensorflow==2.0.0 torch==1.3.1 torchvision==0.4.2</code>
into the terminal (!python and pip need to be installed first, the recommended version for pip is at least 19.3.1). The versions of the modules listed above were used at the time of the creation of these files but future versions of these modules might alos work. Another way to install these packages is by using <code>conda</code>.
## Code
For using the class created for fitting a <code>tf.keras</code> model there are two options:
1. Put the code straight into a python file:<br/>
For that the code from the file [plain.py](plain.py) should be copied into the python file.
2. Importing the class from a different python file:<br/>
For that the file [module.py](module.py) should be inserted into the project folder in which the executed file lies and imported at the top of the executed file:<br/>
<code>from module import DataGenerator</code>
<!---->
In the following python code the following elements should be included:<br/>
```python
  # load the required modules
  import tensorflow.keras as k
  import torch as pt
  from torchvision as tv
  
  # define the transforms for the pytorch dataloader
  # additional transforms from the torch.transforms package can be added
  transform = tv.transforms.Compose(
    [...],
    tv.transforms.ToTensor(),
    [...]
  )
  
  # create the dataloader for the tf.keras model from PyTorch DataLoader object
  dataset = tv.datasets.ImageFolder('path/to/folder', transform=transform)
  dataloader = DataGenerator(pt.utils.data.DataLoader(dataset, [...]), ncl) # ncl represents the number of classes for the model
  
  # creating and defining the tf.keras model
  model = k.models.Sequential()
  [...] # using the model.add([...]) function new layers can be added to the model
  
  model.compile([...]) # compile the model (custom parameter choices)
  model.fit_generator(dataloader, [...]) # fitting the model using the datagenerator (custom parameter choices)
  
  model.save('path/to/model/name.h5') # save the model (optional but useful)
```
The recommended way of using this class is by importing it as a module because docstrings are provided to document the module. In the plain.py file the documentation is not present for shortening the code.

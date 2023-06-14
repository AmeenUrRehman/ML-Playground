#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.feature import Cascade 
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import patches
image = data.astronaut()


# In[2]:


trained_file  = data.lbp_frontal_face_cascade_filename()


# In[3]:


detector = Cascade(trained_file)


# In[4]:


detected = detector.detect_multi_scale(img = image , 
                                      scale_factor =  1.2 , 
                                      step_ratio = 1 ,
                                      min_size  = (10,10) ,
                                      max_size = (200 , 200))
print(detected)


# In[5]:



def show_detected_face(result , detected , title = "Astronaut Image"):
    plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis("off")
    
    for patch in detected:
        img_desc.add_patch(
        patches.Rectangle((patch['c'] , patch['r']) , patch['width'] , patch['height'] , fill= False , color = 'r' , linewidth = 2)
        )
        
        plt.show()


# In[6]:


show_detected_face(image , detected)


# In[ ]:





# In[ ]:





# In[ ]:





# DEW-Model

TensorFlow implementation of the project *When Was This Picture Taken? – Image Date
Estimation in the Wild*.

You can find further information in our paper: [https://link.springer.com/chapter/10.1007/978-3-319-56608-5_57](https://link.springer.com/chapter/10.1007/978-3-319-56608-5_57)

## Inference

To predict the date a picture was taken, you can use our inference script. You can find a pretrained ResNet v1 50 model (7.1 ME) [here](https://github.com/TIB-Visual-Analytics/DEW-Model/releases/download/v1.1/models.tar.gz).

```
python inference.py -m <PATH_TO_MODEL>/model.ckpt-275000 -p <PATH_TO_IMAGE>
```

## Reference

> Müller, E., Springstein, M., & Ewerth, R. (2017, April). “When Was This Picture Taken?”–Image Date Estimation in the Wild. In European Conference on Information Retrieval (pp. 619-625). Springer, Cham.

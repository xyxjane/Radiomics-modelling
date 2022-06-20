'''
Description: 
Author: xueyuxin_jane
Date: 2020-07-28 17:20:42
LastEditTime: 2021-05-16 14:52:22
LastEditors: xueyuxin_jane
'''
from __future__ import print_function
import os
import logging
import SimpleITK as sitk
import radiomics
import yaml 
import six
from radiomics import featureextractor, firstorder, shape, glcm, glrlm, glszm, gldm, ngtdm, imageoperations

# Get some test data

# Download the test case to temporary files and return it's location. If already downloaded, it is not downloaded again,
# but it's location is still returned.
imageName = "../egfr/lung egfr 3mm/PET_series/01.mha"
maskName =  "../egfr/lung egfr 3mm/PET_label/01.mha"
PET_CT_path = "PSMA_75"
parafile_path = "LungNode_Pre/Params.yaml"

def get_features(image, mask, parafile_path):
  f = open(parafile_path,'r',encoding='utf-8')
  para_file = f.read()
  parameters = yaml.load(para_file)
  setting = parameters["setting"]
  # firstorder_list = list(parameters["featureClass"]["firstorder"])
  # shape_list = list(parameters["featureClass"]["shape"])
  glcm_list = list(parameters["featureClass"]["glcm"])

  # Show the first order feature calculations
  firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **setting)
  firstOrderFeatures.enableAllFeatures()
  # Initialize feature extractor
  # for featureName in firstorder_list:
  #   # print(featureName)
  #   firstOrderFeatures.enableFeatureByName(featureName)
  # By default, only original is enabled. Optionally enable some image types:
  # extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
  print("Calculating firstorder features")
  firstOrderFeatures = firstOrderFeatures.execute()

  # Show shape feature calculations
  shapeFeatures =  shape.RadiomicsShape(image, mask, **setting)
  shapeFeatures.enableAllFeatures()
  # Initialize feature extractor
  # for featureName in shape_list:
  #   shapeFeatures.enableFeatureByName(featureName)
  # By default, only original is enabled. Optionally enable some image types:
  # extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
  print("Calculating shape features")
  shapeFeatures = shapeFeatures.execute()

  # Show the glcm feature calculations
  glcmFeatures = glcm.RadiomicsGLCM(image, mask, **setting)
  # glcmFeatures.enableAllFeatures()
  # Initialize feature extractor
  for featureName in glcm_list:
    glcmFeatures.enableFeatureByName(featureName)
  # By default, only original is enabled. Optionally enable some image types:
  # extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
  print("Calculating glcm features")
  glcmFeatures = glcmFeatures.execute()

  # Show the glrlmr feature calculations
  glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **setting)
  # Initialize feature extractor
  glrlmFeatures.enableAllFeatures()
  # By default, only original is enabled. Optionally enable some image types:
  # extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
  print("Calculating glrlm features")
  all_glrlmFeatures = glrlmFeatures.execute()

  # Show the first order feature calculations
  glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **setting)
  # Initialize feature extractor
  glszmFeatures.enableAllFeatures()
  # By default, only original is enabled. Optionally enable some image types:
  # extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
  print("Calculating glszm features")
  all_glszmFeatures = glszmFeatures.execute()

  # Show the first order feature calculations
  gldmFeatures = gldm.RadiomicsGLDM(image, mask, **setting)
  # Initialize feature extractor
  gldmFeatures.enableAllFeatures()
  # By default, only original is enabled. Optionally enable some image types:
  # extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
  print("Calculating gldm features")
  gldmFeatures = gldmFeatures.execute()

  # Show the first order feature calculations
  ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask, **setting)
  # Initialize feature extractor
  ngtdmFeatures.enableAllFeatures()
  # By default, only original is enabled. Optionally enable some image types:
  # extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
  print("Calculating ngtdm features")
  ngtdmFeatures = ngtdmFeatures.execute()

  featureVector_list = [firstOrderFeatures, shapeFeatures, glcmFeatures,all_glrlmFeatures, all_glszmFeatures, gldmFeatures, ngtdmFeatures]
  name_list = ["firstorder", "shape","glcm", "glrlm", "glszm","gldm", "ngtdm"]
  # featureVector_list = [glcmFeatures]
  # name_list = ["glcm"]
  # featureVector_list = [all_glszmFeatures]
  # name_list = ["glszm"]
  # for featureName in featureVector.keys():
  #   print("Computed %s: %s" % (featureName, featureVector[featureName]))
  feature_dic = dict(zip(name_list,featureVector_list))
  return feature_dic

def feature_extraction(imageName, maskName, parafile_path, image_type):
  image = sitk.ReadImage(imageName)
  mask = sitk.ReadImage(maskName)
  # print("printing image",image)
  # print("printing image",mask)
  if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
    print('Error getting testcase!')
    exit()

  # Regulate verbosity with radiomics.verbosity (default verbosity level = WARNING)
  # radiomics.setVerbosity(logging.INFO)

  # Get the PyRadiomics logger (default log-level = INFO)
  logger = radiomics.logger
  logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

  # Set up the handler to write out all log entries to a file
  handler = logging.FileHandler(filename='testLog.txt', mode='w')
  formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  # Get setting parameters
  f = open(parafile_path,'r',encoding='utf-8')
  para_file = f.read()
  parameters = yaml.load(para_file)
  setting = parameters["setting"]
  interpolator = parameters["setting"]["interpolator"]
  resampledPixelSpacing = parameters["setting"]["resampledPixelSpacing"]
  if interpolator is not None and resampledPixelSpacing is not None:
    image, mask = imageoperations.resampleImage(image, mask, **setting)

  bb, correctedMask = imageoperations.checkMask(image, mask, **setting)
  if correctedMask is not None:
    mask = correctedMask
  image, mask = imageoperations.cropToTumorMask(image, mask, bb)

  if image_type == "Original":
    feature_dic = get_features(image, mask, parafile_path)
  
  elif image_type == "wavelet":
    feature_dic = {}
    for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image, mask, **setting):
      single_feature_dic = get_features(decompositionImage, mask, parafile_path)
      temp_dic = {decompositionName:single_feature_dic}
      feature_dic.update(temp_dic)
      print('Calculated firstorder features with wavelet ', decompositionName)
      for (key, val) in six.iteritems(single_feature_dic):
        waveletFeatureName = '%s_%s' % (str(decompositionName), key)
        print(waveletFeatureName)
  else:
    print("wrong type of image")
    feature_dic = None

  return feature_dic

if __name__ == "__main__":
  all_patients = os.listdir(PET_CT_path)
  for patient in all_patients:
      if patient != ".DS_Store" and patient=="1":
        print(patient)
        ct_data_path = os.path.join(PET_CT_path,patient,"PET_SUV.mha")
        ct_label_path = os.path.join(PET_CT_path,patient,"label.mha")
        feature_extraction(ct_data_path,ct_label_path,parafile_path,image_type="Original")

# ct_data_path = os.path.join("/Users/xueyuxin/Documents/research_PET:CT/PSMA_unsorting/10645573_张启华？/CT WB  3.0  B30f.mha")
# ct_label_path = os.path.join("/Users/xueyuxin/Documents/research_PET:CT/PSMA_unsorting/10645573_张启华？/Untitled.mha")
# CT = sitk.Image(sitk.ReadImage(imageName))
# label = sitk.Image(sitk.ReadImage(maskName))
# CT_1 = sitk.Image(sitk.ReadImage(ct_data_path))
# label_1 = sitk.Image(sitk.ReadImage(ct_label_path))
# spacing_ct = CT.GetSize()
# spacing_label = label.GetSize()
# spacing_ct_1 = CT_1.SetOrigin((0,0,0))
# spacing_label_1 = label_1.SetOrigin((0,0,0))
# print("right",spacing_ct,spacing_label)
# print("wrong",spacing_ct_1,spacing_label_1)
# lsif = sitk.LabelStatisticsImageFilter()
# unknown = lsif.Execute(CT_1,label_1)
# print(unknown)



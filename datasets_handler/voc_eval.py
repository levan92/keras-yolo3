# --------------------------------------------------------
# VOC style object detection evaluation, in memory, without using files
# adapted from the original implemention
# mubastan@gmail.com, 06 November 2018
# --------------------------------------------------------

import numpy as np

def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


#def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5):
def voc_eval(detections, annotations, cid, ovthresh=0.1):
  """
  Top level function that does the PASCAL VOC evaluation.
  """
  use_07_metric=False

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  # annotations keyed on image paths
  for imagename in annotations:
    # for each image list of objects: [[x1,y1, x2,y2, cid], [], ...]
    R = [obj[:4] for obj in annotations[imagename] if int(obj[-1]) == cid]
    bbox = np.array(R)
    # difficult is not stored: take it as 0/false
    difficult = np.array([0]*len(R)).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}
  # detections
  image_ids = []
  confidence = []
  BB = []
  # detections should be keyed on class_id (cid)
  class_dict = detections[cid]
  for image_file in class_dict:
      dets = class_dict[image_file]
      for k in range(dets.shape[0]):
          image_ids.append(image_file)
          BB.append(dets[k, 0:4])
          confidence.append(dets[k,-1])
  BB = np.array(BB)
  confidence = np.array(confidence)

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    #sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = 0

  try:
    rec = tp / float(npos)
  except Exception as e:
    print(e)
    # print(imagename)

  # avoid divide by zero in case the first detection matches a difficult ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap

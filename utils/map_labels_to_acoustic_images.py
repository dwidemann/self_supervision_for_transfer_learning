import pandas as pd
import sys, os
import calendar
import csv
import numpy as np
from glob import glob
from datetime import datetime
from datetime import timedelta

month_to_val = {v:k for k,v in enumerate(calendar.month_abbr)}

if __name__ == '__main__':
    labels_file = sys.argv[1]
    
    labels = pd.read_excel(labels_file).to_numpy()
   
    #get a list of labels where the timestamp is converted to datetime and special chars are removed from label field (only happens in some rows of the file read)
    labels = np.apply_along_axis(lambda a: [datetime(a[0].year, a[0].month, a[0].day, a[0].hour, a[0].minute, a[0].second), str(a[1]) + str(a[2])], 1, labels) 

    images_dir = sys.argv[2]

    idx = images_dir.rfind('/') + 1 #index where the filename starts

    fns = glob(os.path.join(images_dir, '*.pkl') )

    outdir = sys.argv[3]
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
   
    idx = images_dir.rfind('/') + 1 #get index of where the file name starts
    
    mapped_labels = [] #resulting mapping between image names and labels
    unique_mapped_labels = [] #only store unique image/label mappings

    output_file_cols = ["label_timestamp", "image_name", "label"]
    unique_output_file_cols = ["image_name", "label"]

    #for every 20 minute acoustic sample, find and map all matching 1 minute sample labels
    for f in fns:
        image_date = f[idx:(idx+12)] #extract image timestamp
        
        #find all labels related to this image (all images represent a 20 minute range, and all labels represent a 1 minute range)
        from_date = datetime(year=2018, month=month_to_val[image_date[:3]], day=int(image_date[3:5]),\
                                hour=int(image_date[6:8]), minute=int(image_date[8:10]), second=int(image_date[10:12]) )
        to_date = from_date + timedelta(minutes=20)

        matching_idxs = (labels[:,0] >= from_date) & (labels[:,0] <= to_date)
        
        if not np.any(matching_idxs): #some images seem don't have labels, print and store which
            print('No label for image:', f[idx:], '  Time:', image_date)    
            
            #store unlabeled mapping in file with all mappings
            mapped_labels.append( [l[0], f[idx:], '0None'] )

            #since image has no label, it should also be added to the unique mappings file
            unique_mapped_labels.append( [f[idx:], '0None'] )
        
        else:#store unique label mappings to image. some images have more than 1 label associated to it. TODO need to determine what to do about this fact
            
            #there should only be 1 or 2 labels per image
            for label in np.unique(labels[matching_idxs, 1]):
                unique_mapped_labels.append([f[idx:], label ])
        

        #if found, store all label mapings to image (even if they are duplicate labels)
        for l in labels[matching_idxs]:
            #(label timestamp, unlabeled image name, label)
            mapped_labels.append( [l[0], f[idx:], l[1]] )
  
    #store image to label mappings in a csv file
    with open(outdir + 'mapped_acoustic_labels.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(output_file_cols)
        writer.writerows(mapped_labels)
  
    with open(outdir + 'unique_mapped_acoustic_labels.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(unique_output_file_cols)
        writer.writerows(unique_mapped_labels)

    print('\nFinished creating label mappings to acoustic images.')

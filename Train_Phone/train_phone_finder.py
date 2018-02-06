import os
import tensorflow as tf
import glob
import csv

#Flags for Input
'''
Example:
python train_phone_finder.py --input_path=find_phone/

'''
flags=tf.app.flags
flags.DEFINE_string('input_path', '','Give path for folder. eg: python train_phone_finder.py --input_path=find_phone/')

FLAGS=flags.FLAGS

def main(_):
    assert FLAGS.input_path, '`input_path` is missing.'
    if FLAGS.input_path:
        input_path=FLAGS.input_path
    
    #Loading the data from text and Converting to CSV
    img_path=[]
    for i in glob.glob(input_path+'/*.jpg'):
        img_path.append(i)
    print("Image paths Loaded")
    
    txt_file = r"find_phone/labels.txt"
    csv_file = r"find_phone/labels.csv"
    
    in_txt = open(txt_file, "r",encoding="utf8").read()
    csvreader=csv.reader(in_txt.splitlines())
    
    row=[]
    for r in csvreader:
        row.append(r[0].split(" "))
   
    with open(csv_file,'w',newline='') as out:
        mywriter=csv.writer(out)
        mywriter.writerow(['Filename','n_xcord','n_ycord'])
        for rr in row:
            mywriter.writerow(rr)
    print("Labels file loaded and converted to csv")
    print("Starting Model Training...")
    
    os.system('python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config')
    
if __name__ == '__main__':
  tf.app.run()

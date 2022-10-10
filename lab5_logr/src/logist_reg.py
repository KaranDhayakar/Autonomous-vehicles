#!/usr/bin/env python
'''
    Logistic Regression
    To solve for a pixel-wise logistic regression model, do: 

    python logist_reg.py trainimname trainmaskname testimname --testmask testmaskname

    Once you have solved for these parameters, you can apply them to an image with
    the apply() function.  This outputs a probability image for pixels in the image.

    Daniel Morris, April 2020
    Copyright 2020
'''
import os
import numpy as np
import argparse
import cv2 as cv
from scipy.special import expit             # Sigmoid function
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression     

def plotClassifcation( img, mask, pixProbs, threshold=0.5, savename='', outfolder='.'):
    ''' Plot Classification Image Results, and output to files '''
    cv.imshow("Image", img)
    cv.imshow("Raw Output", pixProbs )
    if mask.any():
        cv.imshow("Ground Truth Mask", mask)

        #Create colored scoring image:
        TP = np.logical_and( mask > 0, pixProbs > threshold )   # green
        FP = np.logical_and( mask == 0, pixProbs > threshold )  # red
        FN = np.logical_and( mask > 0, pixProbs <= threshold )  # blue
        #add gray color if any of the above labels to reduce contrast slightly
        alabel = TP + FP + FN    
        # R,G,B classification for FP, TP, FN:
        eimg = np.stack( (FN, TP, FP), axis=2 ).astype(np.uint8) * 180 + 75 * alabel[:,:,None].astype(np.uint8)
        # Superimpose this on grayscale image:
        gimg = img.mean(axis=2).astype(np.uint8)
        combimg = (eimg * 3.0/5.0 + gimg[:,:,None] * 2.0/5.0).astype(np.uint8)
        cv.imshow("Scoring Using Mask", combimg)
    if outfolder:
        os.makedirs(outfolder,exist_ok=True)
        cv.imwrite(os.path.join(outfolder,'prob_'+savename), np.uint8(pixProbs*255) )
        if mask.any():
            cv.imwrite(os.path.join(outfolder,'scoring_'+savename), combimg )

def plotTargets(img, target_mask, centroids, savename='', outfolder=''):
    ''' Plot detected target_mask and output to file
        img: image (NxMx3) numpy array
        target_mask: (NxM) numpy array, or else empty list
        centroids: list of [x,y] centroids
    '''
    if isinstance(target_mask,list):
        target_mask = np.array(target_mask)  # Needs to be a numpy array
    if target_mask.size:  # Check if not empty numpy array:
        # Then highlight the detected pixels in the original image
        green = np.zeros_like(img)
        green[:,:,1] = 128
        mask = target_mask[:,:,None].repeat(3,axis=2)
        outim = img.copy() * (1-mask) + (img.copy()//2 + green) * mask
    else:
        outim = img.copy()  
    for centroid in centroids:
        loc = tuple( np.array(centroid).astype(int) )  # location needs to be int tuple
        cv.circle(outim, loc, 5, (0,0,255), -1 )
    cv.imshow("Target", outim)
    if outfolder:
        os.makedirs(outfolder,exist_ok=True)
        cv.imwrite(os.path.join(outfolder,'target_'+savename), outim )

def imread_channel( filename ):
    ''' Read in image and return first channel '''
    img0 = cv.imread( filename )
    if img0 is None:
        print('Warning, unable to read:', filename)
    if len(img0.shape)==3:
        img0 = img0[:,:,0]
    return img0

class LogisticReg:
    def __init__(self ):
        ''' Initialize class with zero values '''
        self.cvec = np.zeros( (1,3) )        
        self.intercept = np.zeros( (1,) )

    def set_model(self, cvec, intercept):
        ''' Set model parameters manually
            cvec:      np.array([[p1,p2,p3]])
            intercept: np.array([p4])
        '''
        self.cvec = cvec
        self.intercept = intercept

    def fit_model_to_files(self, img_name, mask_name, exmask_name='', mod_channels=False):
        ''' Load images from files and fit model parameters '''
        img = cv.imread( img_name )
        mask = imread_channel( mask_name )
        if img is None or mask is None:
            print('Error loading image and mask')
            print('image:', img_name)
            print('mask:', mask_name)
        if exmask_name:
            exmask = imread_channel(exmask_name)
        else:
            exmask = np.array([])
        if mod_channels:
            img = self.modify_img_channels(img)
        self.fit_model( img, mask, exmask )

    def modify_img_channels(self, img):
        # <...> here is where you can modify the image channels.  For now does nothing.
        #img[:][:][3] = img[:][:][4]
        
        grey_sc = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        hue_lig_sat = cv.cvtColor(img,cv.COLOR_BGR2HLS)
        #hue_sat = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        
        luv = cv.cvtColor(img,cv.COLOR_BGR2Luv)
        #lab = cv.cvtColor(img,cv.COLOR_BGR2Lab)

        ret_temp1 = np.concatenate((img,hue_lig_sat[:,:,:]),axis=2)
        ret_temp2 = np.concatenate((ret_temp1,luv[:,:,:]),axis=2)
        ret = np.concatenate((ret_temp2,grey_sc[:,:,None]),axis=2)
        return ret

    def fit_model(self, img, mask, exmask=np.array([]) ):
        ''' Do logistic regression to discriminate points in non-zero region of 
            mask from other points in mask and save estimated logistic regression parameters
            exmask: optionally exclude some pixels from image '''   
        data = img.reshape((-1,img.shape[2])).astype(float)   # Reshape to N x 3
        label = (mask.ravel()>0).astype(int)       # Reshape to length N 
        if exmask.any():                    # Optionally exclude pixels
            keep = exmask.ravel()==0
            data = data[keep,:]
            label = label[keep]
        sk_logr = LogisticRegression(class_weight='balanced',solver='lbfgs')
        sk_logr.fit( data, label)
        self.cvec      = sk_logr.coef_                # Extract coefficients
        self.intercept = np.array(sk_logr.intercept_) # Extract intercept

    def print_params(self):
        print('Logist Regression params, cvec:',self.cvec,'intercept:',self.intercept)

    def apply(self, img, mod_channels=False):
        ''' Application of trained logisitic regression to an image
            img:         [MxNx3] input 3-channel color image
            score:       [MxN] logistic regression score for each pixel (between -inf and inf)
        '''
        if mod_channels:
            img = self.modify_img_channels(img)
        # <...> Solve for the score for each pixel using the parameters estimated in fit_model, namely: self.cvec and self.intercept
        #       Replace the all-zeros assignment below
        #print(img.shape)

        #score = np.zeros( img.shape[0:2] )

        #reshaped_cvec = self.cvec.reshape(-1)

        score = img * self.cvec.reshape(-1)
        score = score.sum(2) + self.intercept

        print(score.shape)
        return score

    def prob_target(self, score):
        ''' Transforms score to probability of target using sigmoid '''
        return expit( score )

    def find_largest_target(self, prob_target, threshold=0.5, minpix=20):
        ''' Finds largest contiguous target region
            This takes the output of logistic regression, thresholds it
            and finds the largest contiguous region and returns its centroid
            Useful for simple cases where you are sure that the target is the
            largest object in the image of its color.
            prob_target: [MxN] input probability of target
            centroid:    [x,y] coordinates of the largest centroid
            area:        number of pixels in target
            target_mask: [MxN] binary mask with 1's at target
        '''
        # Complete this function.  Hint: use cv.connectedComponentsWithStats()

        
        prob_mask = (prob_target>threshold).astype(np.uint8)
        N, label_img, bbn, centroids = cv.connectedComponentsWithStats( prob_mask )

        index = np.argsort(bbn[:,4])

        out_mask = np.zeros_like(prob_mask)

        for i in index[::-1]:
            if prob_mask[label_img==i].astype(float).mean() > 0.99:
                out_mask += (label_img==i).astype(np.uint8)
                centroid = centroids[i,:]
                area = bbn[i,4]
                break

        print(centroid)
        return centroid,area,out_mask

        

        '''
        largest_targ = 0
        largest_targ_index = 0

        prob_mask = np.empty_like(prob_target)
        prob_mask = (prob_target>threshold).astype(np.uint8)
        
        N, label_img, bbn, centroids = cv.connectedComponentsWithStats( prob_mask )
        
        np.set_printoptions(threshold=np.inf)
        #print("here:")
        #print(label_img)#here
        for i in range(N):
            #print(f'{i:3}      {bbn[i,:4]}   {bbn[i,4]:4}    {centroids[i,:]}')
            if (largest_targ < bbn[i,4]):
                largest_targ = bbn[i,4]
                largest_targ_index = i
            
        print("Largest target is of size in pixels",largest_targ)
        print("Largest target's centroid is at",centroids[largest_targ_index,:])
        centroid_out = centroids[largest_targ_index,:]
        area_out = largest_targ
        out_mask = np.zeros_like(prob_mask)
        out_mask += (label_img==largest_targ).astype(np.uint8)

        #print("here:")
        #print(out_mask)#here
        return centroid_out,area_out,out_mask
        #pass

        '''
        

if __name__ == '__main__':
    # This is a demonstration of how LogisticReg can be used
    parser = argparse.ArgumentParser(description='Logistic Regression')
    parser.add_argument('trainimg',      type=str,              help='Train image')
    parser.add_argument('trainmask',     type=str,              help='Train mask')
    parser.add_argument('testimg',       type=str,              help='Test image')
    parser.add_argument('--testmask',    type=str, default='',  help='Test mask')
    parser.add_argument('--trainexmask', type=str, default='',  help='Train pixels to exclude')
    parser.add_argument('--threshold', type=float, default=0.5, help='Target pixel threshold')
    parser.add_argument('--outfolder',   type=str, default='',  help='Folder to save images, if empty do not save')
    parser.add_argument('--mod-channels', dest='mod_channels', action='store_true', help='Modify image channels')
    parser.add_argument('--find-target', dest='find_target', action='store_true', help='Find largest target')
    args = parser.parse_args()

    # Build color model with Logistic Regression
    logr = LogisticReg( )
    logr.fit_model_to_files( args.trainimg, args.trainmask, args.trainexmask, mod_channels=args.mod_channels )

    # Load test data:
    testimg = cv.imread(args.testimg)
    if args.testmask:
        testmask = imread_channel(args.testmask)
    else:
        testmask = np.array([])

    # Apply model to test data:
    score = logr.apply( testimg, mod_channels=args.mod_channels )
    probt = logr.prob_target( score )
    logr.print_params()
    if testmask.size:
        # If we provide ground truth on test data, we can calculate the average precision:
        average_precision = average_precision_score(testmask.ravel()*255, score.ravel() )
        print(f'Average precision: {average_precision:.3f}')

    if args.outfolder:
        print('Saving output images to:',args.outfolder)
    # Plot classification results:
    plotClassifcation(testimg, testmask, probt, args.threshold, os.path.basename(args.testimg), args.outfolder )

    if args.find_target:
        # Find largest regions as targets
        centroid, area, target = logr.find_largest_target(probt, args.threshold)
        # Plot detected region and centroid:
        plotTargets(testimg, target, [centroid], os.path.basename(args.testimg), args.outfolder )

    cv.waitKey()
    cv.destroyAllWindows()


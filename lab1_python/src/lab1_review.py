#!/usr/bin/env python
# coding: utf-8
from scipy.spatial.transform import Rotation as R
from operator import length_hint
from re import A, X
import string
import numpy as np
import cv2 as cv
import os
# Do not import any more packages than the above
'''
    La1 1 Assignment 
    Based on Python Introduction Notes: https://github.com/dmorris0/python_intro

    Complete the following functions by replacing the pass command and/or variables set equal to None
    Functions need to return the specified output.  In most cases only a single line of code is required.  
    To test your functions and get a score out of 20, call:
      python lab1_student_score.py
    Or run lab1_score.py in VSCode.  When you have everything correct, the output should be:
....................
----------------------------------------------------------------------
Ran 20 tests in 0.100s

OK
    Also, you'll see 3 images displayed which you can close.
'''

####################
# Chapter 4: Strings

def find_warning(message: str) -> str:    
    if "warning" in message or "Warning" in message or "WARNING" in message: 
       return 'Warning'
    else:
       return -1
    '''
    Returns the index of the first instance of the substring "warning" or "Warning" (or any other variation on the capitalization)
    If there is no "warning" substring, returns -1
    Hint: don't forget to use a "return" statement
    '''
    pass

def every_third(message: str) -> str:
    i = 2
    x = ''
    while i<len(message):
        x = x + message[i] + ' '
        i = i+1
    return x
    
    '''
    Returns every third letter in message starting with the second letter
    '''
    pass

def all_words(message: str) -> str:
    a = []
    i = 0
    j = 0
    for character in message:
        if message[i] != ' ':
            a.insert(j,message[i])
        else:
            j = j+1
    return a
        
    '''
    Breaks message up at each space (" ") and puts the substrings into a list in the same order
    (Don't worry about punctuation)
    '''
    pass
    
def half_upper_case(message: str) -> str:
    i = 0
    if len(message)%2 == 0:
        while i<len(message):
            if(i<(len(message)/2)):
                message[i] = message[i].upper()
            else :
                message[i] = message[i].lower()
            i = i+1
    else:
        while i<len(message):
            if(i<(len(message)/2)-1):
                message[i] = message[i].upper()
            else :
                message[i] = message[i].lower()
            i = i+1
    return message
        


    '''
    Returns new_message, where new_message has the same letters as message, but the first half
        of the letters are upper case and the rest lower case.  
        If there are an odd number of letters, round down, that is the first half will have one fewer letters
    '''
    pass

#############################
# Chapter 5: Numbers and Math

def c_to_f(degree_c: float) -> float:    
    f = (degree_c*(9/5)) +32
    return f
    '''
    Converts Celcius to Fahrenheit using the formula
    °F = °C * 9/5 + 32 
    Returns output in Fahrenheit
    '''
    pass
    
def exp_div_fun(a: int, b: int, c: int) -> int:
    x = (a*a*b)/c
    return x
    '''
    Return the integer remainder you get when you multiply a times itself b times and then divide by c
    '''
    pass
    
 
 #################################
# Chapter 6: Functions and Loops
    
    
def lcm(x: int, y: int) -> int:
    i = 1
    if x<y :
        i = y
        while True:
            if(i%x == 0):
                return i
            i = i+y
    elif y<x :
        i = x
        while True:
            if(i%y == 0):
                return i
            i = i+x
    else:
        return x

    '''
    
    Return lowest common multiple of x and y
    Method: let m be the larger of x and y
    Let testval start at m and in a loop increment it by m while testval is not divisible by both x and y
    return testval
    Hint: % will be useful
    '''
    pass               

##################################################
# Chapter 8: Conditional Logic and Control Flow

def cond_cum_sum(a: int, b: int) -> int:
    sum=0
    i = 0
    while i<a:
        if i%b != 0:
            sum = sum+i
        i = i +1
    return sum

    '''
    Find the cumulative sum of numbers from 0 to a-1 that are not divisible by b
    Hint: % will be useful
    '''
    pass

def divide_numbers(a: float, b: float) -> float:
    if b != 0:
        x = a/b
        return x
    elif a<0:
        return np.NINF
    elif a>0:
        return np.inf 
    else:
        return 0

    ''' 
    Return a / b
    Perform exception handling for ZeroDivisionError, and in this
    case return signed infinity that is the same sign as a
    Hint: np.sign() and np.inf will be useful
    '''
    pass

##################################################
# Chapter 9: Tuples, Lists and Dictionaries    

def inc_list(a: int, b: int) -> list:
    x = []
    i = a 
    while i<b:
        x.append(i)
        i = i +1
    return x 
    '''
    Return a list of numbers that start at a and increment by 1 to b-1
    '''
    pass

def make_all_lower( string_list: list ) -> list:
    return [string_list.lower() for string_list in string_list]
    ''' Use a single line of Python code for this function
        string_list: list of strings
        returns: list of same strings but all in lower case
        Hint: list comprehension
    '''
    pass

def decrement_string(mystr: str) -> str:
    b = ''
    for k in mystr:
         b += chr(ord(k)-1)
    return b 
    '''
    letter = [x for x in mystr]
    a = []
    i = 0
    while i<len(letter):
        z= ord(letter[i]) - 1
        a[i] = chr(z)
        i = i +1
    y = "".join(a)
    '''
    ''' Use a single line of Python code for this function (hint: list comprehension)
        mystr: a string
        Return a string each of whose characters has is one ASCII value lower than in mystr
        Hint: ord() returns ASCII value, chr() converts ASCII to character, join() combines elements of a list
    '''
    pass

def list2dict( my_list: list ) -> dict:
    x = {}
    i = 0
    for i in my_list:
        x[i] = i * i
    print(x)

    ''' 
    Return a dictionary corresponding to my_list where the keys are elements of my_list
    and the values are the square of the key
    '''
    pass

def concat_tuples( tuple1: tuple, tuple2: tuple ) -> tuple:
    x = tuple1 + tuple2
    return x
    ''' 
    Return a tuple that concatenates tuple2 to the end of tuple1
    '''
    pass


##################################################
# Chapter 13: Numpy 
    
def matrix_multiplication(A: np.array,B: np.array) -> np.array:
    np.matmul(A, B)
    ''' 
    A, B: numpy arrays
    Return: matrix multiplication of A and B
    '''
    pass

def largest_row(M: np.array) -> np.array:
    x = np.argmax(M.sum(axis=1))
    return M[x]

    ''' 
    M: 2D numpy array
    Return: 1D numpy array corresponding to the row with the greatest sum in M
    Hint: use np.argmax
    '''
    pass   

def column_scale( A: np.array, vec: np.array) -> np.array:
    x = A * vec
    return x 
    '''
    A: [M x N] 2D array
    vec: lenth N array
    return [M x N] 2D array where the i'th column is the corresponding column of A * vec[i]
    Hint: use broadcasting to do this in a single line
    '''
    pass

def row_add( A: np.array, vec: np.array) -> np.array:
    x = A + vec
    return x 
    '''
    A: [M x N] 2D array
    vec: lenth M array
    return [M x N] 2D array where the i row is the corresponding row of A + vec[i]
    Hint: use broadcasting to do this in a single line
    '''
    pass

##################################################
# Chapter 14: scipy  

def rotate_90_y(A: np.array) -> np.array:
    r = R.from_euler('XYZ', [0, 90, 0], degrees=True)
    return np.round(r.apply(A))
    '''
    A: [Mx3] array of M 3D points
    return: [Mx3] corresponding to points in A rotated by 90 degrees around Y axis
    Hint: use the scipy rotation operation
    '''
    pass


##################################################
# Chapter 15: OpenCV

class TailLights:

    def __init__(self, impath: str):
        self.impath = impath
        self.img = cv.imread(impath)      
        if self.img is None:
            print('')
            print('*'*60)  # If we can't find the image, check impath
            print('** Current folder is:      ',os.getcwd())
            print('** Unable to read image:   ',impath)  
            print('** Pass correct path to data folder during initialization')
            print('*'*60)
            self.init_succeed = False
        else:
            self.init_succeed = True

    def find_lights_pix(self, show=False) -> np.array:
        #self.img[:,:,2]#red
        #self.img[:,:,0]#blue
        #self.img[:,:,1]#green
        rows,col,bgr = self.img.shape
        out = np.empty((rows, col,1))
        mask_img = self.img.copy()
        #print(mask_img)
        for i in range(len(mask_img)):
            for j in range(len(mask_img[i])):
                if( (mask_img[i][j][0] > 70) and (mask_img[i][j][0] <= 180) and (mask_img[i][j][1] > 120) and (mask_img[i][j][1] <= 190) and (mask_img[i][j][2] > 220) ):
                    #mask_img[i][j][:] = 255
                    out[i][j][0] = 1
                else:
                    #mask_img[i][j][:] = 0
                    out[i][j][0] = 0
        cv.imshow("Lights Pix", out)
        self.img = out
        return self.img
        
        ''' Returns a binary mask image for self.img.  This should be 1 for pixels that meet the following conditions on 
            their red, green and blue channels, and zero otherwise.
            red > 220 AND blue > 70 AND blue <= 180 AND green > 120 AND green <= 190
            Note: do NOT iterate over pixels.  
            Hint: The mask can be created in one line of Python code using multiplication operations.  
                  Think about how to implement an AND operation for arrays of 1s and 0s.
            show: if True, then shows the mask in a window called 'Lights Pix'
        '''
        pass

    def find_lights_bbox(self) -> np.array:    
        N, label_img, bbn, centroids = cv.connectedComponentsWithStats(self.find_lights_pix())
        arr = np.empty((N, 4))
        for i in range(N):
            arr[i][:] = bbn[i]
        print(arr)
        ''' Finds bounding box around lights 
            returns [Mx4] bounding boxes, one for each light.  Each row is [left, top, width, height] in pixels
            Hint: use cv.connectedComponentsWithStats, see Python_Intro documentation
        '''
        pass
"""
    def draw_lights(self, show=False) -> tuple:
        rect = self.img.copy()
        rectangle_list = self.find_lights_bbox()

        for i in range(len(rectangle_list)):
            cv.rectangle(rect, rectangle_list[i][:], color=(0,255,0), thickness=2 )

        cv.imshow('Detected Lights',rect)
        ''' Draw red rectangles with thickness 2 pixels on the image around each detected light
            Returns image with rectangles draw on it.
            show: if True, then displays output image in a window called 'Detected Lights'
        '''
        pass
"""


if __name__=="__main__":

    # A simple way to debug your functions above is to create tests below.  Then
    # you can run this code directly either in VSCode or by typing: 
    # python lab1_review.py

    # For example, you could do
    print( find_warning("Here is a warning") )

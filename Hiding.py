import math
import copy
import random
import cv2
import numpy as np
import multiprocessing
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import math
class Embedding_matric:
    def __init__(self,max_change_valn=9,matrix_size = 256):
        self.max_change_valn = max_change_valn
        self.matrix_size = matrix_size
        self.magic_size = int(math.sqrt(self.max_change_valn))
        self.magic = self.creat_magic()
        self.embedding_matric = None
        self.creat_magic_matrix()


    def creat_magic(self):  # creat n*n magic
        temp = list(range(self.magic_size * self.magic_size))
        random.shuffle(temp)
        matrix = []
        for i in range(self.magic_size):
            matrix.append([])
            for j in range(self.magic_size):
                matrix[i].append(temp[self.magic_size * i + j])
        return matrix

    def creat_magic_matrix(self):  # creat magic matrix
        #stare = time.time()
        matrix_size = math.ceil(self.matrix_size / self.magic_size) + 1

        self.embedding_matric = self.magic

        for i in range(matrix_size-1):
            self.embedding_matric = np.hstack((self.embedding_matric, self.magic))

        temp = self.embedding_matric
        for i in range(matrix_size-1):
            self.embedding_matric = np.vstack((self.embedding_matric, temp))

        # print("creat_magic_matrix time:", time.time() - stare)
        return self.embedding_matric

    def print_magic(self):
        for i in self.magic:
            for j in i:
                print(j, end=" ")
            print()

    def __len__(self):
        return (len(self.embedding_matric),len(self.embedding_matric[0]))

    def __getitem__(self, index):
        x,y = index
        return self.embedding_matric[x][y]

class Hiding_core:
    def __init__(self,parallel = True,base = 4,change_range = [0,2],pix_len = 4):
        self.base = base
        self.parallel = parallel
        self.thread_count = multiprocessing.cpu_count()
        self.Matric = Embedding_matric(self.base,256)
        self.embedding_matric = self.Matric.embedding_matric
        self.secret_dim = None
        self.cover_dim = None
        self.pix_len = pix_len
        self.p = 2 * self.pix_len
        self.change_range = change_range
        if self.parallel:
            print("Using parallel computing Cpu_count:", self.thread_count)
        else:
            print("Didn't use parallel")

    def hiding_message_parallel(self,cover_image,secret_message,key_file = "key.npz"):
        cover_image = np.array(cover_image)
        self.cover_dim = cover_image.shape
        cover_image = cover_image.reshape(-1)

        secret_message = np.array(secret_message)
        self.secret_dim = secret_message.shape
        secret_message = secret_message.reshape(-1)

        #print(len(secret_message), len(cover_image))
        if  len(secret_message)*self.pix_len*2> len(cover_image):
            print("The secret message is too long , please use larger imag")
            return "The secret message is too long"

        stare = time.time()
        if not self.parallel:
            stego_image = self.hiding_message(cover_image,secret_message).reshape(self.cover_dim)
        else:
            #print("Using parallel computing Cpu_count:",self.thread_count)
            secret_message_block_size = len(secret_message) // self.thread_count#計算分割後每塊影像大小
            process = []
            p = Pool(self.thread_count)#定義值行緒行緒
            for i in range(self.thread_count):
                if i != self.thread_count-1:
                    process.append(p.apply_async(self.hiding_message,
                        args = (cover_image[i*secret_message_block_size*self.p:i*secret_message_block_size*self.p + secret_message_block_size*self.p],
                        secret_message[i*secret_message_block_size : i*secret_message_block_size + secret_message_block_size])))#將影像分割後放入不同值行緒執行
                else:
                    process.append(p.apply_async(self.hiding_message,
                        args=(cover_image[i * secret_message_block_size * self.p:],
                        secret_message[i * secret_message_block_size:])))#將影像分割後放入不同值行緒執行
            p.close()
            p.join()#等待所有值行緒執行完畢

            stego_image = np.array([], dtype=np.uint8)
            for i in range(self.thread_count):
                stego_image = np.concatenate([stego_image,process[i].get()])#將各值行緒結果合併再在一起
        # self.save_key(key_file)
        # print("Save key file in",key_file)
        print("hiding time:", time.time() - stare)
        return stego_image.reshape(self.cover_dim)

    def find_val(self,i,j,val):
        #(x,y)
        #print(val)
        for i_p in range(self.change_range[0],self.change_range[1]):
            for j_p in range(self.change_range[0],self.change_range[1]):
                if self.embedding_matric[i+i_p][j+j_p] == val:
                    return i+i_p,j+j_p
        #print(val)

    def hiding_message(self,cover_image,secret_message):
        secret_message = self.base_conversion(secret_message)
        secret_message = self.secret_message_to_secret_string(secret_message)
        i = 0
        try:
            for iteam in secret_message:
                cover_image[i], cover_image[i + 1] = self.find_val(cover_image[i], cover_image[i + 1], int(iteam))
                i = i + 2
        except Exception as e:
            print(e)
            return e
        return cover_image

    def base_conversion(self,secret_message):
        conversion_result = []
        for i in range(len(secret_message)):
            res = ""
            val = secret_message[i]
            while True:
                Quotient = val // self.base
                remainder = val % self.base
                res = str(remainder)+res
                val = Quotient
                if val < self.base:
                    res = str(val) + res
                    break
            conversion_result.append(res)
        return conversion_result

    def base_reconversion(self,secret_message):
        reconversion_result = []
        for i in range(0,len(secret_message),self.pix_len):
            temp = 0
            for j in range(self.pix_len):
                temp += int(secret_message[i+j])*pow(self.base,self.pix_len-j-1)
            reconversion_result.append(temp)
        return reconversion_result

    def secret_message_to_secret_string(self,secret_message):
        secret_string = ""
        for iteam in secret_message:
            temp = str(iteam)
            #print(temp,iteam)
            for i in range(len(temp),self.pix_len):
                temp = '0' + temp
            secret_string += temp

        return secret_string

    def extracting_message_parallel(self,cover_image,key_file = "key.npz"):
        self.Matric.magic,self.secret_dim = self.load_key(key_file)
        self.embedding_matric = self.Matric.creat_magic_matrix()

        cover_image = np.array(cover_image)
        self.cover_dim = cover_image.shape
        cover_image = cover_image.reshape(-1)

        size = self.secret_dim[0]*self.secret_dim[1]*self.secret_dim[2]
        stare = time.time()
        if not self.parallel:
            secret_message = self.extracting_message(cover_image,size,self.embedding_matric).reshape(self.secret_dim)
        else:
            #print("Using parallel computing Cpu_count:", self.thread_count)
            secret_message_block_size = size // self.thread_count
            process = []
            secret_message = np.array([], dtype=np.uint8)
            p = Pool(self.thread_count)
            for i in range(self.thread_count):
                if i != self.thread_count - 1:
                    process.append(p.apply_async(self.extracting_message,
                                                 args=(cover_image[
                                                       i * secret_message_block_size * self.p:i * secret_message_block_size * self.p + secret_message_block_size * self.p],
                                                       secret_message_block_size,self.embedding_matric)))
                else:
                    process.append(p.apply_async(self.extracting_message,
                                                 args=(cover_image[i * secret_message_block_size * self.p:],
                                                       size-(secret_message_block_size*(self.thread_count-1)),self.embedding_matric)))
            p.close()
            p.join()

            for i in range(self.thread_count):
                # print(process[i].get())
                secret_message = np.concatenate([secret_message, process[i].get()])

        print("extracting time:", time.time() - stare)
        return secret_message.reshape(self.secret_dim)

    def extracting_message(self,cover_image,size,embedding_matric):
        try:
            temp = []
            for i in range(size*self.pix_len):
                temp.append(str(embedding_matric[cover_image[2*i]][cover_image[2*i+1]]))
            temp = self.base_reconversion(temp)
            secret_message = temp
            secret_message = np.array(secret_message,dtype = np.uint8)
            return secret_message
        except Exception as e:
            print(e)
            return e

    def save_key(self,key_file = "key.npz"):
        embedding_matric = np.array(self.Matric.magic) #將嵌入矩陣轉成numpy格式(4*4)
        size = np.array(self.secret_dim) #將機密影像大小轉成numpy格式
        np.savez(key_file, embedding_matric=embedding_matric, size=size) #儲存key

    def load_key(self,key_file = "key.npz"):
        key = np.load(key_file)
        return key["embedding_matric"] , key["size"]

    def caculate_bpp(self):
        #print(bin(self.base-1))
        return (len(str(bin(self.base-1))[2:]))/2

    def caculate_PSNR(self,org,chang):
        mse = chang-org
        mse = np.sum(mse**2)/(org.shape[0]*org.shape[1]*org.shape[2])
        PSNR = 10*math.log(255**2/mse,10)
        return PSNR

    def caculate_IPHR(self):
        IPHR = 1/(self.pix_len*2)
        return IPHR

class Hiding_quaternary_method(Hiding_core):
    def __init__(self,parallel = True):
        super().__init__(parallel, 4, [0, 2], 4)

class Hiding_Lin_method(Hiding_core):
    def __init__(self,parallel = True):
        super().__init__(parallel, 9, [-1, 2], 3)

class Hiding_modify_method(Hiding_core):
    def __init__(self,parallel = True):
        super().__init__(parallel, 16, [-1, 3], 2)
        self.hex2dec = {"0": 0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"A":10,"B":11,"C":12,"D":13,"E":14,"F":15}
        self.dec2hex = dict(zip(self.hex2dec.values(),self.hex2dec.keys()))

    def hiding_message(self,cover_image,secret_message):
        secret_message = self.base_conversion(secret_message)
        secret_message = self.secret_message_to_secret_string(secret_message)
        i = 0
        try:
            for iteam in secret_message:
                cover_image[i], cover_image[i + 1] = self.find_val(cover_image[i], cover_image[i + 1], self.hex2dec[iteam])
                i = i + 2
        except Exception as e:
            print(e)
            return e
        return cover_image

    def base_conversion(self,secret_message):
        temp = []
        for i in range(len(secret_message)):
            res = ""
            val = secret_message[i]
            while True:
                Quotient = val // 16
                remainder = val % 16
                res = self.dec2hex[remainder]+res
                val = Quotient
                if val < 16:
                    res = self.dec2hex[val] + res
                    break
            temp.append(res)
        return temp

    def base_reconversion(self,secret_message):
        temp = []
        for i in range(0,len(secret_message),2):
            temp.append(int(secret_message[i])*16 + int(secret_message[i+1]))
        return temp

class Hiding_base25_method(Hiding_core):
    def __init__(self,parallel = True):
        super().__init__(parallel, 25, [-2, 3], 2)
        self.base225 = {"0": 0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"A":10,
                        "B":11,"C":12,"D":13,"E":14,"F":15,"G":16,"H":17,"I":18,"J":19,"K":20
                        ,"L":21,"M":22,"N":23,"O":24}
        self.dec2base25 = dict(zip(self.base225.values(),self.base225.keys()))

    def hiding_message(self,cover_image,secret_message):
        secret_message = self.base_conversion(secret_message)
        secret_message = self.secret_message_to_secret_string(secret_message)
        i = 0
        try:
            for iteam in secret_message:
                cover_image[i], cover_image[i + 1] = self.find_val(cover_image[i], cover_image[i + 1], self.base225[iteam])
                i = i + 2
        except Exception as e:
            print(e)
            return e
        return cover_image

    def base_conversion(self,secret_message):
        temp = []
        for i in range(len(secret_message)):
            res = ""
            val = secret_message[i]
            while True:
                Quotient = val // 25
                remainder = val % 25
                res = self.dec2base25[remainder]+res
                val = Quotient
                if val < 25:
                    res = self.dec2base25[val] + res
                    break
            temp.append(res)
        return temp

    def base_reconversion(self,secret_message):
        temp = []
        for i in range(0,len(secret_message),2):
            temp.append(int(secret_message[i])*25 + int(secret_message[i+1]))
        return temp

# class Hiding_quaternary_method:
#     def __init__(self,parallel = True):
#         self.parallel = parallel
#         self.thread_count = multiprocessing.cpu_count()
#         self.Matric = Embedding_matric(4,256)
#         self.embedding_matric = self.Matric.embedding_matric
#         self.secret_dim = None
#         self.cover_dim = None
#         self.p = 2*4
#
#     def hiding_message_parallel(self,cover_image,secret_message,key_file = "key.npz"):
#         cover_image = np.array(cover_image)
#         self.cover_dim = cover_image.shape
#         cover_image = cover_image.reshape(-1)
#
#         secret_message = np.array(secret_message)
#         self.secret_dim = secret_message.shape
#         secret_message = secret_message.reshape(-1)
#
#         print(len(secret_message), len(cover_image))
#         if  len(secret_message)*self.p> len(cover_image):
#             print("The secret message is too long , please use larger imag")
#             return "The secret message is too long"
#
#
#         if not self.parallel:
#             return self.hiding_message(cover_image,secret_message).reshape(self.cover_dim)
#         else:
#             print("Using parallel computing Cpu_count:",self.thread_count)
#             secret_message_block_size = len(secret_message) // self.thread_count
#             process = []
#             stego_image = np.array([],dtype=np.uint8)
#             p = Pool(self.thread_count)
#             for i in range(self.thread_count):
#                 if i != self.thread_count-1:
#                     process.append(p.apply_async(self.hiding_message,
#                         args = (cover_image[i*secret_message_block_size*self.p:i*secret_message_block_size*self.p + secret_message_block_size*self.p],
#                         secret_message[i*secret_message_block_size : i*secret_message_block_size + secret_message_block_size])))
#                 else:
#                     process.append(p.apply_async(self.hiding_message,
#                         args=(cover_image[i * secret_message_block_size * self.p:],
#                         secret_message[i * secret_message_block_size:])))
#             p.close()
#             p.join()
#
#             for i in range(self.thread_count):
#                 stego_image = np.concatenate([stego_image,process[i].get()])
#             self.save_key(key_file)
#             print("Save key file in",key_file)
#             return stego_image.reshape(self.cover_dim)
#
#     def find_val(self,i,j,val):
#         #(x,y)
#         #print(val)
#         for i_p in range(0,2):
#             for j_p in range(0,2):
#                 if self.embedding_matric[i+i_p][j+j_p] == val:
#                     return i+i_p,j+j_p
#
#     def hiding_message(self,cover_image,secret_message):
#         secret_message = self.Decimal_to_quaternary(secret_message)
#         secret_message = self.secret_message_to_secret_string(secret_message)
#         i = 0
#         try:
#             for iteam in secret_message:
#                 cover_image[i], cover_image[i + 1] = self.find_val(cover_image[i], cover_image[i + 1], int(iteam))
#                 i = i + 2
#         except Exception as e:
#             print(e)
#             return e
#         return cover_image
#
#     def Decimal_to_quaternary(self,secret_message):
#         temp = []
#         for i in range(len(secret_message)):
#             res = ""
#             val = secret_message[i]
#             while True:
#                 Quotient = val // 4
#                 remainder = val % 4
#                 res = str(remainder)+res
#                 val = Quotient
#                 if val < 4:
#                     res = str(val) + res
#                     break
#             temp.append(res)
#         return temp
#
#     def quaternary_to_Decimal(self,secret_message):
#         temp = []
#         for i in range(0,len(secret_message),4):
#             temp.append(int(secret_message[i])*64 + int(secret_message[i+1])*16 + int(secret_message[i+2])*4 +  int(secret_message[i+3]))
#         return temp
#
#     def secret_message_to_secret_string(self,secret_message):
#         secret_string = ""
#         for iteam in secret_message:
#             temp = str(iteam)
#             #print(temp,iteam)
#             for i in range(len(temp),4):
#                 temp = '0' + temp
#             secret_string += temp
#
#         return secret_string
#
#     def extracting_message_parallel(self,cover_image,key_file = "key.npz"):
#         self.Matric.magic,self.secret_dim = self.load_key(key_file)
#         self.embedding_matric=self.Matric.creat_magic_matrix()
#         cover_image = np.array(cover_image)
#         self.cover_dim = cover_image.shape
#         cover_image = cover_image.reshape(-1)
#
#         size = self.secret_dim[0]*self.secret_dim[1]*self.secret_dim[2]
#         if not self.parallel:
#             return self.extracting_message(cover_image,size,embedding_matric).reshape(self.secret_dim)
#         else:
#             print("Using parallel computing Cpu_count:", self.thread_count)
#             secret_message_block_size = size // self.thread_count
#             process = []
#             secret_message = np.array([], dtype=np.uint8)
#             p = Pool(self.thread_count)
#             for i in range(self.thread_count):
#                 if i != self.thread_count - 1:
#                     process.append(p.apply_async(self.extracting_message,
#                                                  args=(cover_image[
#                                                        i * secret_message_block_size * self.p:i * secret_message_block_size * self.p + secret_message_block_size * self.p],
#                                                        secret_message_block_size,self.embedding_matric)))
#                 else:
#                     process.append(p.apply_async(self.extracting_message,
#                                                  args=(cover_image[i * secret_message_block_size * self.p:],
#                                                        size-(secret_message_block_size*(self.thread_count-1)),self.embedding_matric)))
#             p.close()
#             p.join()
#
#             for i in range(self.thread_count):
#                 # print(process[i].get())
#                 secret_message = np.concatenate([secret_message, process[i].get()])
#             return secret_message.reshape(self.secret_dim)
#
#     def extracting_message(self,cover_image,size,embedding_matric):
#         try:
#             temp = []
#             for i in range(size*4):
#                 temp.append(str(embedding_matric[cover_image[2*i]][cover_image[2*i+1]]))
#             temp = self.quaternary_to_Decimal(temp)
#             secret_message = temp
#             secret_message = np.array(secret_message,dtype = np.uint8)
#             return secret_message
#         except Exception as e:
#             print(e)
#             return e
#
#     def save_key(self,key_file = "key.npz"):
#         embedding_matric = np.array(self.Matric.magic)
#         size = np.array(self.secret_dim)
#         np.savez(key_file, embedding_matric=embedding_matric, size=size)
#
#     def load_key(self,key_file = "key.npz"):
#         key = np.load(key_file)
#         return key["embedding_matric"] , key["size"]
#
# class Hiding_Kim_method:
#     def __init__(self,parallel = True):
#         self.parallel = parallel
#         self.thread_count = multiprocessing.cpu_count()
#         self.Matric = Embedding_matric(9,256)
#         self.embedding_matric = self.Matric.embedding_matric
#         self.secret_dim = None
#         self.cover_dim = None
#
#     def hiding_message_parallel(self,cover_image,secret_message,key_file = "key.npz"):
#         cover_image = np.array(cover_image)
#         self.cover_dim = cover_image.shape
#         cover_image = cover_image.reshape(-1)
#
#         secret_message = np.array(secret_message)
#         self.secret_dim = secret_message.shape
#         secret_message = secret_message.reshape(-1)
#
#         print(len(secret_message), len(cover_image))
#         if  len(secret_message)*6> len(cover_image):
#             print("The secret message is too long , please use larger imag")
#             return "The secret message is too long"
#
#
#         if not self.parallel:
#             return self.hiding_message(cover_image,secret_message).reshape(self.cover_dim)
#         else:
#             print("Using parallel computing Cpu_count:",self.thread_count)
#             secret_message_block_size = len(secret_message) // self.thread_count
#             process = []
#             stego_image = np.array([],dtype=np.uint8)
#             p = Pool(self.thread_count)
#             for i in range(self.thread_count):
#                 if i != self.thread_count-1:
#                     process.append(p.apply_async(self.hiding_message,
#                         args = (cover_image[i*secret_message_block_size*6:i*secret_message_block_size*6 + secret_message_block_size*6],
#                         secret_message[i*secret_message_block_size : i*secret_message_block_size + secret_message_block_size])))
#                 else:
#                     process.append(p.apply_async(self.hiding_message,
#                         args=(cover_image[i * secret_message_block_size * 6:],
#                         secret_message[i * secret_message_block_size:])))
#             p.close()
#             p.join()
#
#             for i in range(self.thread_count):
#                 stego_image = np.concatenate([stego_image,process[i].get()])
#             self.save_key(key_file)
#             print("Save key file in",key_file)
#             return stego_image.reshape(self.cover_dim)
#
#     def find_val(self,i,j,val):
#         #(x,y)
#         #print(val)
#         for i_p in range(-1,2):
#             for j_p in range(-1,2):
#                 if self.embedding_matric[i+i_p][j+j_p] == val:
#                     return i+i_p,j+j_p
#         print(val)
#
#     def hiding_message(self,cover_image,secret_message):
#         secret_message = self.Decimal_to_novenary(secret_message)
#         secret_message = self.secret_message_to_secret_string(secret_message)
#         i = 0
#         try:
#             for iteam in secret_message:
#                 cover_image[i], cover_image[i + 1] = self.find_val(cover_image[i], cover_image[i + 1], int(iteam))
#                 i = i + 2
#         except Exception as e:
#             print(e)
#             return e
#         return cover_image
#
#     def Decimal_to_novenary(self,secret_message):
#         temp = []
#         for i in range(len(secret_message)):
#             res = ""
#             val = secret_message[i]
#             while True:
#                 Quotient = val // 9
#                 remainder = val % 9
#                 res = str(remainder)+res
#                 val = Quotient
#                 if val < 9:
#                     res = str(val) + res
#                     break
#             temp.append(res)
#         return temp
#
#     def novenary_to_Decimal(self,secret_message):
#         temp = []
#         for i in range(0,len(secret_message),3):
#             temp.append(int(secret_message[i])*81 + int(secret_message[i+1])*9 + int(secret_message[i+2]))
#         return temp
#
#     def secret_message_to_secret_string(self,secret_message):
#         secret_string = ""
#         for iteam in secret_message:
#             temp = str(iteam)
#             #print(temp,iteam)
#             for i in range(len(temp),3):
#                 temp = '0' + temp
#             secret_string += temp
#
#         return secret_string
#
#     def extracting_message_parallel(self,cover_image,key_file = "key.npz"):
#         self.Matric.magic,self.secret_dim = self.load_key(key_file)
#         self.embedding_matric = self.Matric.creat_magic_matrix()
#
#         cover_image = np.array(cover_image)
#         self.cover_dim = cover_image.shape
#         cover_image = cover_image.reshape(-1)
#
#         size = self.secret_dim[0]*self.secret_dim[1]*self.secret_dim[2]
#         if not self.parallel:
#             return self.extracting_message(cover_image,size,embedding_matric).reshape(self.secret_dim)
#         else:
#             print("Using parallel computing Cpu_count:", self.thread_count)
#             secret_message_block_size = size // self.thread_count
#             process = []
#             secret_message = np.array([], dtype=np.uint8)
#             p = Pool(self.thread_count)
#             for i in range(self.thread_count):
#                 if i != self.thread_count - 1:
#                     process.append(p.apply_async(self.extracting_message,
#                                                  args=(cover_image[
#                                                        i * secret_message_block_size * 6:i * secret_message_block_size * 6 + secret_message_block_size * 6],
#                                                        secret_message_block_size,self.embedding_matric)))
#                 else:
#                     process.append(p.apply_async(self.extracting_message,
#                                                  args=(cover_image[i * secret_message_block_size * 6:],
#                                                        size-(secret_message_block_size*(self.thread_count-1)),self.embedding_matric)))
#             p.close()
#             p.join()
#
#             for i in range(self.thread_count):
#                 # print(process[i].get())
#                 secret_message = np.concatenate([secret_message, process[i].get()])
#             return secret_message.reshape(self.secret_dim)
#
#     def extracting_message(self,cover_image,size,embedding_matric):
#         try:
#             temp = []
#             for i in range(size*3):
#                 temp.append(str(embedding_matric[cover_image[2*i]][cover_image[2*i+1]]))
#             temp = self.novenary_to_Decimal(temp)
#             secret_message = temp
#             secret_message = np.array(secret_message,dtype = np.uint8)
#             return secret_message
#         except Exception as e:
#             print(e)
#             return e
#
#     def save_key(self,key_file = "key.npz"):
#         embedding_matric = np.array(self.Matric.magic)
#         size = np.array(self.secret_dim)
#         np.savez(key_file, embedding_matric=embedding_matric, size=size)
#
#     def load_key(self,key_file = "key.npz"):
#         key = np.load(key_file)
#         return key["embedding_matric"] , key["size"]
#
# class Hiding_modify_method:
#     def __init__(self,parallel = True):
#         self.parallel = parallel
#         self.thread_count = multiprocessing.cpu_count()
#         self.Matric = Embedding_matric(16,256+1)
#         self.embedding_matric = self.Matric.embedding_matric
#         self.secret_dim = None
#         self.cover_dim = None
#         self.multiple = 2*2
#         self.hex2dec = {"0": 0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"A":10,"B":11,"C":12,"D":13,"E":14,"F":15}
#         self.dec2hex = dict(zip(self.hex2dec.values(),self.hex2dec.keys()))
#
#     def hiding_message_parallel(self,cover_image,secret_message,key_file = "key.npz"):
#         cover_image = np.array(cover_image)
#         self.cover_dim = cover_image.shape
#         cover_image = cover_image.reshape(-1)
#
#         secret_message = np.array(secret_message)
#         self.secret_dim = secret_message.shape
#         secret_message = secret_message.reshape(-1)
#         print(len(secret_message),len(cover_image))
#         if  len(secret_message)*self.multiple> len(cover_image):
#             print("The secret message is too long , please use larger imag")
#             return "The secret message is too long"
#
#
#         if not self.parallel:
#             return self.hiding_message(cover_image,secret_message).reshape(self.cover_dim)
#         else:
#             print("Using parallel computing Cpu_count:",self.thread_count)
#             secret_message_block_size = len(secret_message) // self.thread_count
#             process = []
#             stego_image = np.array([],dtype=np.uint8)
#             p = Pool(self.thread_count)
#             for i in range(self.thread_count):
#                 if i != self.thread_count-1:
#                     process.append(p.apply_async(self.hiding_message,
#                         args = (cover_image[i*secret_message_block_size*self.multiple:i*secret_message_block_size*self.multiple + secret_message_block_size*self.multiple],
#                         secret_message[i*secret_message_block_size : i*secret_message_block_size + secret_message_block_size])))
#                 else:
#                     process.append(p.apply_async(self.hiding_message,
#                         args=(cover_image[i * secret_message_block_size * self.multiple:],
#                         secret_message[i * secret_message_block_size:])))
#             p.close()
#             p.join()
#
#             for i in range(self.thread_count):
#                 #print(process[i].get())
#                 stego_image = np.concatenate([stego_image,process[i].get()])
#             self.save_key(key_file)
#             print("Save key file in",key_file)
#             return stego_image.reshape(self.cover_dim)
#
#     def find_val(self,i,j,val):
#         #(x,y)
#         try:
#             for i_p in range(-1,3):
#                 for j_p in range(-1,3):
#                     if self.embedding_matric[i+i_p][j+j_p] == val:
#                         return i+i_p,j+j_p
#         except Exception as e:
#             print("find_val error:",e)
#
#     def hiding_message(self,cover_image,secret_message):
#         secret_message = self.Decimal_to_hex(secret_message)
#         secret_message = self.secret_message_to_secret_string(secret_message)
#         i = 0
#         try:
#             for iteam in secret_message:
#                 cover_image[i], cover_image[i + 1] = self.find_val(cover_image[i], cover_image[i + 1], self.hex2dec[iteam])
#                 i = i + 2
#         except Exception as e:
#             print(e)
#             return e
#         return cover_image
#
#     def Decimal_to_hex(self,secret_message):
#         temp = []
#         for i in range(len(secret_message)):
#             res = ""
#             val = secret_message[i]
#             while True:
#                 Quotient = val // 16
#                 remainder = val % 16
#                 res = self.dec2hex[remainder]+res
#                 val = Quotient
#                 if val < 16:
#                     res = self.dec2hex[val] + res
#                     break
#             temp.append(res)
#         return temp
#
#     def hex_to_Decimal(self,secret_message):
#         temp = []
#         for i in range(0,len(secret_message),2):
#             temp.append(int(secret_message[i])*16 + int(secret_message[i+1]))
#         return temp
#
#     def secret_message_to_secret_string(self,secret_message):
#         secret_string = ""
#         for iteam in secret_message:
#             temp = str(iteam)
#             #print(temp,iteam)
#             for i in range(len(temp),2):
#                 temp = '0' + temp
#             secret_string += temp
#
#         return secret_string
#
#     def extracting_message_parallel(self,cover_image,key_file = "key.npz"):
#         self.Matric.magic , self.secret_dim = self.load_key(key_file)
#         self.embedding_matric = self.Matric.creat_magic_matrix()
#
#         cover_image = np.array(cover_image)
#         self.cover_dim = cover_image.shape
#         cover_image = cover_image.reshape(-1)
#
#         size = self.secret_dim[0]*self.secret_dim[1]*self.secret_dim[2]
#         if not self.parallel:
#             return self.extracting_message(cover_image,size,embedding_matric).reshape(self.secret_dim)
#         else:
#             print("Using parallel computing Cpu_count:", self.thread_count)
#             secret_message_block_size = size // self.thread_count
#             process = []
#             secret_message = np.array([], dtype=np.uint8)
#             p = Pool(self.thread_count)
#             for i in range(self.thread_count):
#                 if i != self.thread_count - 1:
#                     process.append(p.apply_async(self.extracting_message,
#                                                  args=(cover_image[
#                                                        i * secret_message_block_size * self.multiple:i * secret_message_block_size * self.multiple + secret_message_block_size * self.multiple],
#                                                        secret_message_block_size,self.embedding_matric)))
#                 else:
#                     process.append(p.apply_async(self.extracting_message,
#                                                  args=(cover_image[i * secret_message_block_size * self.multiple:],
#                                                        size-(secret_message_block_size*(self.thread_count-1)),self.embedding_matric)))
#             p.close()
#             p.join()
#
#             for i in range(self.thread_count):
#                 # print(process[i].get())
#                 secret_message = np.concatenate([secret_message, process[i].get()])
#             return secret_message.reshape(self.secret_dim)
#
#     def extracting_message(self,cover_image,size,embedding_matric):
#         try:
#             temp = []
#             for i in range(size*2):
#                 temp.append(str(embedding_matric[cover_image[2*i]][cover_image[2*i+1]]))
#             temp = self.hex_to_Decimal(temp)
#             secret_message = temp
#             secret_message = np.array(secret_message,dtype = np.uint8)
#             return secret_message
#         except Exception as e:
#             print(e)
#             return e
#
#     def save_key(self,key_file = "key.npz"):
#         embedding_matric = np.array(self.Matric.magic)
#         size = np.array(self.secret_dim)
#         np.savez(key_file, embedding_matric=embedding_matric, size=size)
#
#     def load_key(self,key_file = "key.npz"):
#         key = np.load(key_file)
#         print(key["embedding_matric"])
#         return key["embedding_matric"] , key["size"]

if __name__ == '__main__':

    cover_image = cv2.imread("image.png")
    secret_message = cv2.imread("image.jpg")
    cv2.imshow("cover_image",cover_image)
    cv2.imshow("secret_message", secret_message)
    stare = time.time()
    hiding_system = Hiding_modify_method(True)

    print("init time:", time.time() - stare)

    stare = time.time()
    stego_image = hiding_system.hiding_message_parallel(cover_image,secret_message)
    print("hiding time:",time.time() - stare)
    print(f"bpp:{hiding_system.caculate_bpp()}")
    print(f"PSNR:{hiding_system.caculate_PSNR(cover_image, stego_image)}")
    print(f"IPHR:{hiding_system.caculate_IPHR()}")
    try:
        cv2.imwrite("stego_image.png", stego_image)
        # print(stego_image[:,:,0])
        stego_image = cv2.imread("stego_image.png")
        # print(stego_image[:, :, 0])
        cv2.imshow("stego_image",stego_image)
        stare = time.time()
        extracting_img = hiding_system.extracting_message_parallel(stego_image,"key.npz")
        print("extracting time:", time.time() - stare)
        cv2.imshow("extracting_image", extracting_img)
    except Exception as e:
        print(e)
    cv2.waitKey()
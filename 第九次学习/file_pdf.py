# f =open(r"D:\pycharmcode\第九次学习\3.pdf",mode="rb")
# content = f.read()
# f.close()

"""
# PyPDF2  提取PDF文件中的某几页
# """
# import PyPDF2
# with open("3.pdf","rb") as f:
#     f_0 = PyPDF2.PdfFileReader(f)
#     f_0_w=PyPDF2.PdfFileWriter()
#     for page in range(9,12):
#         page_object= f_0.getPage(page)
#         f_0_w.addPage(page_object)
#         with open("提取9-11页.pdf","wb") as file:
#              f_0_w.write(file)

# 合并PDF文件
# import PyPDF2
# filenames = ["提取2-4页.pdf","提取5-8页.pdf","提取9-11页.pdf"]
# merger = PyPDF2.PdfFileMerger()
# for filename in filenames:
#     merger.append(PyPDF2.PdfFileReader(filename))
# merger.write("合并2-11页的PDF文件.pdf")

# 对PDF文件进行加密
# import PyPDF2
# with open("3.pdf","rb") as f:
#      f_read = PyPDF2.PdfFileReader(f)
#      f_write= PyPDF2.PdfFileWriter()
#      for page in range(f_read.numPages):
#               f_write.addPage(f_read.getPage(page))
#      f_write.encrypt("zhaokai")
#      with open("加密文件.pdf","wb") as e_f:
#          f_write.write((e_f))




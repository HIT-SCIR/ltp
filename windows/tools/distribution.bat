
rmdir /S /Q ..\\share-package\\cplusplus\\test_vs2003\\release
rmdir /S /Q ..\\share-package\\cplusplus\\test_vs2003\\debug
rmdir /S /Q ..\\share-package\\cplusplus\\test_vs2008\\release
rmdir /S /Q ..\\share-package\\cplusplus\\test_vs2008\\debug


del /F /Q ..\\share-package\\cplusplus\\test_vs2003\\*.xml
del /F /Q ..\\share-package\\cplusplus\\test_vs2003\\*.txt
del /F /Q ..\\share-package\\cplusplus\\test_vs2003\\*.user
del /F /Q ..\\share-package\\cplusplus\\test_vs2003\\*.ncb

del /F /Q ..\\share-package\\cplusplus\\test_vs2008\\*.xml
del /F /Q ..\\share-package\\cplusplus\\test_vs2008\\*.txt
del /F /Q ..\\share-package\\cplusplus\\test_vs2008\\*.user
del /F /Q ..\\share-package\\cplusplus\\test_vs2008\\*.ncb

del /F /Q ..\\share-package\\python\\test\\*.xml
del /F /Q ..\\share-package\\python\\test\\*.txt

#mkdir..\\share-package\\cplusplus\\test_vs2003\\src
#mkdir..\\share-package\\cplusplus\\test_vs2003\\src\\__util
#mkdir..\\share-package\\cplusplus\\test_vs2003\\src\\__ltp_dll
#mkdir..\\share-package\\cplusplus\\test_vs2003\\src\\test_suit

mkdir..\\share-package\\cplusplus\\test_vs2008\\src
mkdir..\\share-package\\cplusplus\\test_vs2008\\src\\__util
mkdir..\\share-package\\cplusplus\\test_vs2008\\src\\__ltp_dll
mkdir..\\share-package\\cplusplus\\test_vs2008\\src\\test_suit

#copy /Y ..\\src\\__ltp_dll\\__ltp_dll.h ..\\share-package\\cplusplus\\test_vs2003\\src\\__ltp_dll\\
copy /Y ..\\src\\__ltp_dll\\__ltp_dll.h ..\\share-package\\cplusplus\\test_vs2008\\src\\__ltp_dll\\
#copy /Y ..\\src\\__ltp_dll\\__ltp_dll_x.cpp ..\\share-package\\cplusplus\\test_vs2003\\src\\__ltp_dll\\
copy /Y ..\\src\\__ltp_dll\\__ltp_dll_x.cpp ..\\share-package\\cplusplus\\test_vs2008\\src\\__ltp_dll\\
#copy /Y ..\\src\\__util\\MyLib.* ..\\share-package\\cplusplus\\test_vs2003\\src\\__util\\
copy /Y ..\\src\\__util\\MyLib.* ..\\share-package\\cplusplus\\test_vs2008\\src\\__util\\
#copy /Y ..\\src\\test_suit\\test_ltp_dll.cpp ..\\share-package\\cplusplus\\test_vs2003\\src\\test_suit\\
copy /Y ..\\src\\test_suit\\test_ltp_dll.cpp ..\\share-package\\cplusplus\\test_vs2008\\src\\test_suit\\

#copy /Y ..\\win_lib\\vc71\\release\\__ltp_dll.lib ..\\share-package\\cplusplus\\test_vs2003\\
copy /Y ..\\win_bin\\vc71\\release\\*.dll ..\\share-package\\cplusplus\\test_vs2003\\

#copy /Y ..\\win_lib\\vs2008\\release\\__ltp_dll.lib ..\\share-package\\cplusplus\\test_vs2008\\
copy /Y ..\\win_bin\\vs2008\\release\\*.dll ..\\share-package\\cplusplus\\test_vs2008\\

copy /Y ..\\win_bin\\vs2008\\release\\*.dll ..\\share-package\\python\\test
copy /Y ..\\win_bin\\vs2008\\release\\*.py ..\\share-package\\python\\test

#copy /Y ..\\share-package\\python\\test\\*.conf ..\\share-package\\cplusplus\\test_vs2003\\
copy /Y ..\\share-package\\python\\test\\*.conf ..\\share-package\\cplusplus\\test_vs2008\\

#copy /Y ..\\win_bin\\vs2008\\release\\*.txt ..\\share-package\\cplusplus\\test_vs2003\\
copy /Y ..\\win_bin\\vs2008\\release\\*.txt ..\\share-package\\cplusplus\\test_vs2008\\
copy /Y ..\\win_bin\\vs2008\\release\\*.txt ..\\share-package\\python\\test

copy /Y ..\\doc\\LTP π”√Œƒµµv2.1.pdf ..\\share-package\\


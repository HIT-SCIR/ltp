#!/usr/bin/perl  
$app = ltp_server;  
while(1){  
    $result =  `ps aux | grep $app | grep  -v  grep`;  
    unless($result){  
        system("nohup ./$app &") == 0 
            or die "can not start $app: $!";  
    }  
    sleep 10;  
}   

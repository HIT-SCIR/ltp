#!/usr/bin/perl

use LWP::UserAgent;

my $ua = new LWP::UserAgent;
my $doc = "";

while(<>){
    chomp($_);
    $doc = $doc.$_;
    if($_ eq ""){
        print $doc;
        my $response = $ua->post('http://202.118.250.16:12345/ltp', {t=>'dp', s=>$doc});
        if ($response->is_success) {
            my $result = $response->decoded_content();
            print $result;
        }
        $doc = "";
    }
}

package COMMON;

#Copyright (C) 2004 Jesus Gimenez and Lluis Marquez

#This library is free software; you can redistribute it and/or
#modify it under the terms of the GNU Lesser General Public
#License as published by the Free Software Foundation; either
#version 2.1 of the License, or (at your option) any later version.

#This library is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#Lesser General Public License for more details.

#You should have received a copy of the GNU Lesser General Public
#License along with this library; if not, write to the Free Software
#Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

use strict;
use IO;

use vars qw($appname $appversion $appyear $clsfext $smplext $trainext $valext $testext $mapext $unkext $svmext $SVMEXT $DICTEXT $EXPNDEXT $TAGSEXT $MRGEXT $mrgext $M0EXT $M1EXT $M2EXT $M3EXT $M4EXT $Wext $Bext $FOLDEXT $OUTEXT $DSFEXT $AMBPEXT $UNKPEXT $A0EXT $A1EXT $A2EXT $A3EXT $A4EXT $revext $lrext $rlext $lrmode $rlmode $lrlmode $glrlmode $mode0 $mode1 $mode2 $mode3 $mode4 $st0 $st1 $st2 $st3 $st4 $st5 $st6 $softmax0 $softmax1 $verbose0 $verbose1 $verbose2 $verbose3 $verbose4 $attvalseparator $innerseparator $valseparator $pairseparator $emptypos $unkpos $emptyword $d_separator $in_valseparator $out_valseparator $progress0 $progress1 $progress2 $progress3 $unkamb $WINEXT $IGNORE $SMARK $VERSION);

# --------------------                 #application name and version
$COMMON::appname = "SVMTool";
$COMMON::VERSION = "1.3";
$COMMON::appversion = "1.3";
$COMMON::appyear = "2006";
# --------------------                 #file extensions
$COMMON::clsfext = "CLSF";
$COMMON::smplext = "SAMPLES";
$COMMON::DSFEXT = "DSF";
$COMMON::trainext = "TRAIN";
$COMMON::valext = "VAL";
$COMMON::testext = "TEST";
$COMMON::mapext = "MAP";
$COMMON::knext = "KN";
$COMMON::unkext = "UNK";
$COMMON::svmext = "svm";
$COMMON::SVMEXT = "SVM";
$COMMON::Wext = "W";
$COMMON::Bext = "B";
$COMMON::DICTEXT = "DICT";
$COMMON::EXPNDEXT = "XPND";
$COMMON::TAGSEXT = "TAGS";
$COMMON::MRGEXT = "MRG";
$COMMON::mrgext = "mrg";
$COMMON::EXPEXT = "EXP";
$COMMON::revext = "REV";
$COMMON::lrext = "LR";
$COMMON::rlext = "RL";
$COMMON::MORFEXT = "MORFO";
$COMMON::M0EXT = "M0";
$COMMON::M1EXT = "M1";
$COMMON::M2EXT = "M2";
$COMMON::M3EXT = "M3";
$COMMON::M4EXT = "M4";
$COMMON::FOLDEXT = "FLD";
$COMMON::OUTEXT = "OUT";
$COMMON::AMBPEXT = "AMBP";
$COMMON::UNKPEXT = "UNKP";
$COMMON::A0EXT = "A0";
$COMMON::A1EXT = "A1";
$COMMON::A2EXT = "A2";
$COMMON::A3EXT = "A3";
$COMMON::A4EXT = "A4";
$COMMON::WINEXT = "WIN";
# --------------------                 #tagging directions
$COMMON::lrmode = "LR";
$COMMON::rlmode = "RL";
$COMMON::lrlmode = "LRL";
$COMMON::glrlmode = "GLRL";
# --------------------                 #SVM model modes
$COMMON::mode0 = 0;
$COMMON::mode1 = 1;
$COMMON::mode2 = 2;
$COMMON::mode3 = 3;
$COMMON::mode4 = 4;
# --------------------                 #tagging strategies
$COMMON::st0 = 0;
$COMMON::st1 = 1;
$COMMON::st2 = 2;
$COMMON::st3 = 3;
$COMMON::st4 = 4;
$COMMON::st5 = 5;
$COMMON::st6 = 6;
# --------------------                 #softmax functions
$COMMON::softmax0 = 0;
$COMMON::softmax1 = 1;
# --------------------                 #verbosity level
$COMMON::verbose0 = 0;
$COMMON::verbose1 = 1;
$COMMON::verbose2 = 2;
$COMMON::verbose3 = 3;
$COMMON::verbose4 = 4;
# --------------------
$COMMON::in_valseparator = " ";        #input separator
$COMMON::out_valseparator = " ";       #output separator
$COMMON::pairseparator = " ";          #pair separator
$COMMON::attvalseparator = ":";        #attribute-value separator
$COMMON::innerseparator = "_";         #value internal separator
$COMMON::d_separator = " ";            #dictionary separator
$COMMON::valseparator = "~";           #value separator         (cuidado porque joden el split de perl)
$COMMON::emptyword = "_";              #empty item
$COMMON::emptypos = "??";              #empty pos
$COMMON::unkpos = "*?";                #unknown pos
$COMMON::unkamb = "UNKNOWN";           #unknown ambiguity class
# --------------------                 #ignore_line start character
$COMMON::IGNORE = "##";
$COMMON::SMARK = "<s>";
# --------------------                 #progress values
$COMMON::progress0 = 100;
$COMMON::progress1 = 1000;
$COMMON::progress2 = 10000;
$COMMON::progress3 = 10;

# --------------------------------------------------------------------------------

sub report
{
   #description _ reports a give string onto STDOUT and FILE
   #param1 _ report filename
   #param2 _ string

   my $report = shift;
   my $string = shift;

   print $string;
   my $REPORT = new IO::File(">> $report") or die "Couldn't open report file: $report\n";
   print $REPORT $string;
   $REPORT->close();
}

sub end_of_sentence
{
    #description _ returns true iif the given word is a candidate to end a sentence
    #param1 _ given word

    my $word = shift;

    if(($word eq "¡£") or ($word eq "£¿") or ($word eq "£¡")){
		return 1;
		}
	  else 
	  {return 0;}
}

sub print_time
{
    #description _ given a time in seconds prints it in a HOUR:MIN:SEC
    #param1 _ time in seconds

    my $nseconds = shift;

    my $hours = int($nseconds / 3600);
    my $minutes = int(($nseconds % 3600) / 60);
    my $seconds = $nseconds % 60;
    print "REAL TIME: $hours hrs : $minutes min : $seconds sec\n";
}

sub print_time_file
{
    #description _ given a time in seconds prints it in a HOUR:MIN:SEC
    #param1 _ time in seconds
    #param2 _ output file handle

    my $nseconds = shift;
    my $FILE = shift;

    my $hours = int($nseconds / 3600);
    my $minutes = int(($nseconds % 3600) / 60);
    my $seconds = $nseconds % 60;
    print "TIME: $hours hrs : $minutes min : $seconds sec\n";
    print $FILE "TIME: $hours hrs : $minutes min : $seconds sec\n";
}

sub print_benchmark
{
    #description _ prints the benchmark
    #param1  _ first benchmark
    #param2  _ second benchmark
    #@return _ time difference

    my $time1 = shift;
    my $time2 = shift;

    #print STDERR "BENCHMARK TIME (1): ", Benchmark::timestr($time1) ,"\n";
    #print STDERR "BENCHMARK TIME (2): ", Benchmark::timestr($time2) ,"\n";
    printf STDERR "BENCHMARK TIME: %.4f\n", Benchmark::timestr(Benchmark::timediff($time2,$time1));

    return (Benchmark::timediff($time2, $time1))->[1];
}

sub get_benchmark
{
    #description _ returns the benchmark time
    #param1  _ first benchmark
    #param2  _ second benchmark
    #@return _ time difference

    my $time1 = shift;
    my $time2 = shift;

    return (Benchmark::timediff($time2, $time1))->[1];
}

sub print_benchmark_file
{
    #description _ prints the benchmark
    #param1 _ first benchmark
    #param2 _ second benchmark
    #param3 _ output file handle

    my $time1 = shift;
    my $time2 = shift;
    my $file = shift;

    my $FILE = new IO::File(">> $file") or die "Couldn't open report file: $file\n";

    #print "BENCHMARK TIME (1): ", Benchmark::timestr($time1) ,"\n";
    #print "BENCHMARK TIME (2): ", Benchmark::timestr($time2) ,"\n";
    printf "BENCHMARK TIME: %.4f\n", Benchmark::timestr(Benchmark::timediff($time2,$time1));
    #print $FILE "BENCHMARK TIME (1): ", Benchmark::timestr($time1) ,"\n";
    #print $FILE "BENCHMARK TIME (2): ", Benchmark::timestr($time2) ,"\n";
    printf $FILE "BENCHMARK TIME: %.4f\n", Benchmark::timestr(Benchmark::timediff($time2,$time1));

    $FILE->close();
}

sub compute_accuracy
{
    #description _ given nhits and nsamples, computes accuracy
    #param1 _ nhits
    #param2 _ nsamples

    my $hits = shift;
    my $nsamples = shift;

    my $accuracy = 0;
    if ($nsamples != 0) { $accuracy = $hits / $nsamples; }

    return $accuracy;
}

sub print_accuracy
{
    #description _ given nhits and nsamples, computes accuracy and then prints it onto STDOUT
    #param1 _ nhits
    #param2 _ nsamples

    my $hits = shift;
    my $nsamples = shift;

    my $accuracy = compute_accuracy($hits, $nsamples);
    my $acc = sprintf("%.4f", $accuracy * 100) + 0;

    print "ACCURACY = $hits / $nsamples = $acc \%\n";

    return $accuracy;
}

sub print_acc
{
    #description _ given nhits and nsamples, computes accuracy and then prints it onto STDOUT
    #param1 _ nhits
    #param2 _ nsamples

    my $hits = shift;
    my $nsamples = shift;

    my $accuracy = compute_accuracy($hits, $nsamples);
    my $acc = sprintf("%.4f", $accuracy * 100) + 0;

    print "$acc";

    return $accuracy;
}

sub print_accuracy_file
{
    #description _ given nhits and nsamples, computes accuracy
    #param1 _ nhits
    #param2 _ nsamples
    #param3 _ output file handle

    my $hits = shift;
    my $nsamples = shift;
    my $REPORT = shift;

    my $accuracy = 0;

    if ($nsamples != 0) { $accuracy = $hits / $nsamples; }

    print "ACCURACY = $hits / $nsamples = $accuracy\n";

    print $REPORT "ACCURACY = $hits / $nsamples = $accuracy\n";

    return $accuracy;
}

sub do_hash
{
    #description _ responsible for generating a hash out from a <att:val> list
    #param1 _ list
    #param2 _ <key>:<value> separator ('_' usually)
    #         ignored if empty -> every list item is a key
    #                             values are set to 1 by default (boolean true)
    #@return value -> hash reference

    my $rlist = shift;
    my $separator = shift;

    my %hash;

    my $len = @{$rlist};
    my $i = 0;
    my @item;

    while ($i < $len) {
       if ($separator ne "") {
         @item = split(/$separator/, $rlist->[$i]);
         $hash{$item[0]} = $item[1];
       }
       else {
	   $hash{$rlist->[$i]} = 1;
       }
       $i++;
    }

    return \%hash;
}

sub randomize{
    #description _ Randomly distributes a corpus into 3 sets...
    #              (training 60%; validation 20%; test 20%)
    #param1 _ input corpus filename
    #param2 _ training set filename
    #param3 _ validation set filename
    #param4 _ test set filename
    #param5 _ percentage for training   (60%)
    #param6 _ percentage for validation (20%)
    #param7 _ percentage for test       (20%)

    my $smplset = shift;
    my $trainset = shift;
    my $valset = shift;
    my $testset = shift;
    my $Xtrain = shift;
    my $Xval = shift;
    my $Xtest = shift;

    my $CORPUS = new IO::File("< $smplset") or die "Couldn't open input file: $smplset\n";
    my $TRAIN = new IO::File("> $trainset") or die "Couldn't open output file: $trainset\n";
    my $VAL = new IO::File("> $valset") or die "Couldn't open output file: $valset\n";
    my $TEST = new IO::File("> $testset") or die "Couldn't open output file: $testset\n";

    print "RANDOMLY SPLITTNG THE CORPUS...<$smplset> INTO\n";
    my $pctrain = $Xtrain / ($Xtrain + $Xval + $Xtest);
    my $pcval = $Xval / ($Xtrain + $Xval + $Xtest);
    my $pctest = $Xtest / ($Xtrain + $Xval + $Xtest);
    print "<$trainset> - $pctrain% -\n<$valset> - $pcval% -\n<$pctest> - 20% -\n";

    srand();
    my $rn;

    my $ntrain = 0;
    my $nval = 0;
    my $ntest = 0;

    while (defined(my $line = $CORPUS->getline())) {
        $rn = rand ($Xtrain + $Xval + $Xtest);
        if ($rn < $Xtrain) { print $TRAIN $line; $ntrain++; }
        elsif (($rn >= $Xtrain) && ($rn < $Xtrain + $Xval)) { print $VAL $line; $nval++; } 
	elsif (($rn >= $Xtrain + $Xval) && ($rn < $Xtrain + $Xval + $Xtest)) { print $TEST $line; $ntest++; }
    }

    $CORPUS->close();
    $TRAIN->close();
    $VAL->close();
    $TEST->close();
}

sub create_folders{
    #description _ Randomly splits a corpus into equal size folders
    #param1 _ input corpus filename
    #param2 _ folded corpus filename
    #param3 _ number of folders
    #param4 _ report file
    #param5 _ verbosity

    my $corpus = shift;
    my $folded = shift;
    my $nfolders = shift;
    my $report = shift;
    my $verbose = shift;

    my $CORPUS = new IO::File("< $corpus") or die "Couldn't open input file: $corpus\n";
    my $FOLDED = new IO::File("> $folded") or die "Couldn't open output file: $folded\n";

    if ($verbose > $COMMON::verbose1) { COMMON::report($report, "CREATING $nfolders FOLDERS from <$corpus> into <$folded>\n"); }

    srand();
    my $rn;
    $rn = rand ($nfolders);
    my $sentence = "";
    while (defined(my $line = $CORPUS->getline())) {
        chomp($line);
        $sentence .= sprintf("%d", $rn).$COMMON::in_valseparator.$line."\n";
        my @item = split($COMMON::in_valseparator, $line);
        if (end_of_sentence($item[0])) {
	   print $FOLDED $sentence;
           $sentence = "";
           $rn = rand ($nfolders);
        }
    }

    $CORPUS->close();
    $FOLDED->close();
}

sub pick_up_folders{
    #description _ Randomly splits a corpus into equal size folders
    #param1 _ folded corpus filename
    #param2 _ train corpus filename
    #param3 _ test corpus filename
    #param4 _ folder number
    #param5 _ report file
    #param6 _ verbosity

    my $folded = shift;
    my $train = shift;
    my $test = shift;
    my $f = shift;
    my $report = shift;
    my $verbose = shift;

    my $FOLDED = new IO::File("< $folded") or die "Couldn't open input file: $folded\n";
    my $TRAIN = new IO::File("> $train") or die "Couldn't open output file: $train\n";
    my $TEST = new IO::File("> $test") or die "Couldn't open output file: $test\n";

    if ($verbose > $COMMON::verbose1) { COMMON::report($report, "\n---------------------------------------------------------------------------------------------\nFOLDER $f\n---------------------------------------------------------------------------------------------\nPICKING UP <$train> and <$test> FOLDERS from $folded\n"); }

    while (defined(my $line = $FOLDED->getline())) {
        chomp($line);
        my @entry = split(/$COMMON::in_valseparator/, $line);
        my $linef = shift(@entry);
        my $new = join("$COMMON::in_valseparator", @entry);
        if ($linef eq $f) { print $TEST "$new\n"; }
        else { print $TRAIN "$new\n"; }
    }

    $FOLDED->close();
    $TRAIN->close();
    $TEST->close();
}

sub show_progress_stdout
{
    #description _ prints progress bar onto standard error output
    #param1 _ iteration number
    #param2 _ print "." after N p1 iterations
    #param3 _ print "#iter" after N p2 iterations

    my $iter = shift;
    my $p1 = shift;
    my $p2 = shift;

    if (($iter % $p1) == 0) { print "."; }
    if (($iter % $p2) == 0) { print "$iter"; }
}

sub show_progress_stderr
{
    #description _ prints progress bar onto standard error output
    #param1 _ iteration number
    #param2 _ print "." after N p1 iterations
    #param3 _ print "#iter" after N p2 iterations

    my $iter = shift;
    my $p1 = shift;
    my $p2 = shift;

    if (($iter % $p1) == 0) { print STDERR "."; }
    if (($iter % $p2) == 0) { print STDERR "$iter"; }
}

sub show_progress
{
    #description _ prints progress bar onto standard error output
    #param1 _ iteration number
    #param2 _ print "." after N p1 iterations
    #param3 _ print "#iter" after N p2 iterations

    my $iter = shift;
    my $p1 = shift;
    my $p2 = shift;

    if (($iter % $p1) == 0) { print STDERR "."; }
    if (($iter % $p2) == 0) { print STDERR "$iter"; }
}

sub reverse_list
{
    #description _ responsible for inverting the order of the elements inside a list
    #param1  _ input list
    #@return _ reverted list

    my $input = shift;
 
    my @output;
    my $i = scalar(@{$input});
    while ($i > 0) {
	push(@output, $input->[$i-1]);
       $i--;
    }

    return \@output;
}

sub write_list
{
    #description _ sorts and writes a list onto a file
    #param1 _ list reference
    #param2 _ filename
    #param3 _ verbosity

    my $list = shift;
    my $file = shift;
    my $verbose = shift;

    my $FILE = new IO::File("> $file") or die "Couldn't open output file: $file\n";

    if ($verbose > 0) { print STDERR "STORING <$file>\n"; }
    foreach my $elem (@{$list}) {
       print $FILE "$elem\n";
    }

    $FILE->close();
}

sub read_list
{
    #description _ reads a list from a file
    #param1 _ filename

    my $file = shift;

    my @list;

    my $FILE = new IO::File("< $file") or die "Couldn't open input file: $file\n";

    while (defined(my $line = $FILE->getline())) {
       chomp($line);
       push(@list, $line);
    }

    $FILE->close();

    return(\@list);
}


sub reverse_file
{
    #description _ reverses a file (equivalent to the 'tac' command)
    #param1 _ input filename
    #param2 _ output filename

    my $infile = shift;
    my $outfile = shift;

    #system ("tac $infile > $outfile");

    my $IN = new IO::File("< $infile") or die "Couldn't open input file: $infile\n";
    my $OUT = new IO::File("> $outfile") or die "Couldn't open output file: $outfile\n";

    my @content;
    while (defined(my $line = $IN->getline())) {
       unshift(@content, $line);
    }

    foreach my $line (@content) {
       print $OUT $line;
    }

    $IN->close();
    $OUT->close();
}

1;

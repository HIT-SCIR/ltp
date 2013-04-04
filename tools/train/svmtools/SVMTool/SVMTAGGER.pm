package SVMTAGGER;

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
use Benchmark;
use Data::Dumper;
use SVMTool::COMMON;
use SVMTool::DICTIONARY;
use SVMTool::SWINDOW;
use SVMTool::SVM;
use SVMTool::MAPPING;
use SVMTool::ATTGEN;
use SVMTool::ENTRY;
use SVMTool::STATS;

# -----------------------------------------------------------------------------------
# local variables
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
#DEFAULT VALUES

# -- SVM
$SVMTAGGER::MIN_VALUE = -99999999;
$SVMTAGGER::Ck = 0;
$SVMTAGGER::Cu = 0;

# -- X
$SVMTAGGER::X = 3;
# -- Derr
$SVMTAGGER::dicterr = 0.001;             #heuristic dictionary
# -- EQerr
$SVMTAGGER::eqerr = 0.01;                #classes of equivalence
# -- Kfilter
$SVMTAGGER::Kfilter = 0;                 #weight filtering for known words
# -- Ufilter
$SVMTAGGER::Ufilter = 0;                 #weight filtering for unknown words

# -- sliding window --
$SVMTAGGER::wlength = 5;
$SVMTAGGER::wcorepos = 2;

#tagging modes
$SVMTAGGER::Mdefault = 0;
$SVMTAGGER::Mspecial = 1;
$SVMTAGGER::Mviterbi = 2;

#viterbi
$SVMTAGGER::ENDCTX = "END";

#remove intermediate files
$SVMTAGGER::rmfiles = 1;

#remake folders for cross-validation
$SVMTAGGER::remakeFLDS = 1;

#corpus splitting probabilities
$SVMTAGGER::TRAINP = 0.8;
$SVMTAGGER::TESTP = 0.1;
$SVMTAGGER::VALP = 0.1;

#cross-validation number of folders
$SVMTAGGER::NFOLDERS = 10;

# -----------------------------------------------------------------------------------
#ambiguous-right [default]
my @lATT0k = ("w(-2)", "w(-1)", "w(0)", "w(1)", "w(2)", "w(-2,-1)", "w(-1,0)", "w(0,1)", "w(-1,1)", "w(1,2)", "w(-2,-1,0)", "w(-2,-1,1)", "w(-1,0,1)", "w(-1,1,2)", "w(0,1,2)", "p(-2)", "p(-1)", "p(-2,-1)", "p(-1,1)", "p(1,2)", "p(-2,-1,1)", "p(-1,1,2)", "k(0)", "k(1)", "k(2)", "m(0)", "m(1)", "m(2)");
$SVMTAGGER::rATT0k = \@lATT0k;
my @lATT0u = ("w(-2)", "w(-1)", "w(0)", "w(1)", "w(2)", "w(-2,-1)", "w(-1,0)", "w(0,1)", "w(-1,1)", "w(1,2)", "w(-2,-1,0)", "w(-2,-1,1)", "w(-1,0,1)", "w(-1,1,2)", "w(0,1,2)", "p(-2)", "p(-1)", "p(-2,-1)", "p(-1,1)", "p(1,2)", "p(-2,-1,1)", "p(-1,1,2)", "k(0)", "k(1)", "k(2)", "m(0)", "m(1)", "m(2)", "a(2)", "a(3)", "a(4)", "z(2)", "z(3)", "z(4)", "ca(1)", "cz(1)", "L", "SA", "AA", "SN", "CA", "CAA", "CP", "CC", "CN", "MW");
$SVMTAGGER::rATT0u = \@lATT0u;
# -----------------------------------------------------------------------------------
#unambiguous-right
my @lATT1k = ("w(-2)", "w(-1)", "w(0)", "w(1)", "w(2)", "w(-2,-1)", "w(-1,0)", "w(0,1)", "w(-1,1)", "w(1,2)", "w(-2,-1,0)", "w(-2,-1,1)", "w(-1,0,1)", "w(-1,1,2)", "w(0,1,2)", "p(-2)", "p(-1)", "p(1)", "p(2)", "p(-2,-1)", "p(-1,0)", "p(-1,1)", "p(0,1)", "p(1,2)", "p(-2,-1,0)", "p(-2,-1,1)", "p(-1,0,1)", "p(-1,1,2)", "k(0)", "k(1)", "k(2)", "m(0)", "m(1)", "m(2)");
$SVMTAGGER::rATT1k = \@lATT1k;
my @lATT1u = ("w(-2)", "w(-1)", "w(0)", "w(1)", "w(2)", "w(-2,-1)", "w(-1,0)", "w(0,1)", "w(-1,1)", "w(1,2)", "w(-2,-1,0)", "w(-2,-1,1)", "w(-1,0,1)", "w(-1,1,2)", "w(0,1,2)", "p(-2)", "p(-1)", "p(1)", "p(2)", "p(-2,-1)", "p(-1,0)", "p(-1,1)", "p(0,1)", "p(1,2)", "p(-2,-1,0)", "p(-2,-1,1)", "p(-1,0,1)", "p(-1,1,2)", "k(0)", "k(1)", "k(2)", "m(0)", "m(1)", "m(2)", "a(2)", "a(3)", "a(4)", "z(2)", "z(3)", "z(4)", "ca(1)", "cz(1)", "L", "SA", "AA", "SN", "CA", "CAA", "CP", "CC", "CN", "MW");
$SVMTAGGER::rATT1u = \@lATT1u;
# -----------------------------------------------------------------------------------
#no-right
my @lATT2k = ("w(-2)", "w(-1)", "w(0)", "w(1)", "w(2)", "w(-2,-1)", "w(-1,0)", "w(0,1)", "w(-1,1)", "w(1,2)", "w(-2,-1,0)", "w(-2,-1,1)", "w(-1,0,1)", "w(-1,1,2)", "w(0,1,2)", "p(-2)", "p(-1)", "p(-2,-1)", "k(0)", "m(0)");
$SVMTAGGER::rATT2k = \@lATT2k;
my @lATT2u = ("w(-2)", "w(-1)", "w(0)", "w(1)", "w(2)", "w(-2,-1)", "w(-1,0)", "w(0,1)", "w(-1,1)", "w(1,2)", "w(-2,-1,0)", "w(-2,-1,1)", "w(-1,0,1)", "w(-1,1,2)", "w(0,1,2)", "p(-2)", "p(-1)", "p(-2,-1)", "k(0)", "m(0)", "a(2)", "a(3)", "a(4)", "z(2)", "z(3)", "z(4)", "ca(1)", "cz(1)", "L", "SA", "AA", "SN", "CA", "CAA", "CP", "CC", "CN", "MW");
$SVMTAGGER::rATT2u = \@lATT2u;
# -----------------------------------------------------------------------------------
#unsupervised-learning
my @lATT3k = ("w(-2)", "w(-1)", "w(0)", "w(1)", "w(2)", "w(-2,-1)", "w(-1,0)", "w(0,1)", "w(-1,1)", "w(1,2)", "w(-2,-1,0)", "w(-2,-1,1)", "w(-1,0,1)", "w(-1,1,2)", "w(0,1,2)", "p(-2)", "p(-1)", "p(-2,-1)", "p(-1,1)", "p(1,2)", "p(-2,-1,1)", "p(-1,1,2)", "k(-2)", "k(-1)", "k(1)", "k(2)", "m(-2)", "m(-1)", "m(1)", "m(2)");
$SVMTAGGER::rATT3k = \@lATT3k;
my @lATT3u = ("w(-2)", "w(-1)", "w(0)", "w(1)", "w(2)", "w(-2,-1)", "w(-1,0)", "w(0,1)", "w(-1,1)", "w(1,2)", "w(-2,-1,0)", "w(-2,-1,1)", "w(-1,0,1)", "w(-1,1,2)", "w(0,1,2)", "p(-2)", "p(-1)", "p(-2,-1)", "p(-1,1)", "p(1,2)", "p(-2,-1,1)", "p(-1,1,2)", "k(-2)", "k(-1)", "k(1)", "k(2)", "m(-2)", "m(-1)", "m(1)", "m(2)", "a(2)", "a(3)", "a(4)", "z(2)", "z(3)", "z(4)", "ca(1)", "cz(1)", "L", "SA", "AA", "SN", "CA", "CAA", "CP", "CC", "CN", "MW");
$SVMTAGGER::rATT3u = \@lATT3u;
# -----------------------------------------------------------------------------------

$SVMTAGGER::rSTRATS = { $COMMON::mode0 => $COMMON::st0, $COMMON::mode1 => $COMMON::st1, $COMMON::mode2 => $COMMON::st1, $COMMON::mode3 => $COMMON::st3, $COMMON::mode4 => $COMMON::st4};
$SVMTAGGER::rMEXTS = { $COMMON::mode0 => $COMMON::M0EXT, $COMMON::mode1 => $COMMON::M1EXT, $COMMON::mode2 => $COMMON::M2EXT, $COMMON::mode3 => $COMMON::M3EXT, $COMMON::mode4 => $COMMON::M4EXT};

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------


# =============================================================================================
# =============================== PRIVATE METHODS =============================================
# =============================================================================================

# =================================== GENERAL =========================================
sub read_unihan
{
	my $input = "result.TXT";
	open  INF, "<$input" or die "open file err: $input\n";
  
	my %Unihan;
	while (<INF>) {
		chomp;
		my @line = split(/ /,$_);
		$Unihan{$line[1]} = $line[0];
	}	
	close INF;
	return \%Unihan;
}

sub read_bs
{
	my $input = "bs.TXT";
	open  INF, "<$input" or die "open file err: $input\n";
  
  my @BS;
	my $i=0;
	while (<INF>) {
		chomp;
		my @line = split(/ /,$_);
		$BS[$i] = $line[0];
		$i++;
	}	
	close INF;
 return \@BS;
}

sub read_fs_list
{
    #description _ conveniently reads and transforms a feature set list
    #param1  _ list
    #param2  _ window setup
    #@return _ feature set list

    my $fslist = shift;
    my $wlen = shift;
    my $wcore = shift;

    my @fs;
    my $minpos = 0 - $wcore;
    my $maxpos = $wlen - $wcore - 1;
    foreach my $f (@{$fslist}) {
       my @action = split(/[()]/, $f);
       my @args;
       my $column;
       if ($action[0] eq "C") { #MULTIPLE-COLUMNS
          my @ARGS = split(/[;]/, $action[1]);
          $column = $ARGS[0];
          @args = split(/[,]/, $ARGS[1]);
       }
       else {
          @args = split(/[,]/, $action[1]);
       }
       my $valid = 1;
       if (ATTGEN::check_arguments($action[0])) {
          if (scalar(@args) == 0) { die "[WRONG FEATURE!!] a range must be specified for feature $f\n"; }
          foreach my $a (@args) {
             if (!(($a >= $minpos) and ($a <= $maxpos))) {
                die "[WRONG FEATURE!!] invalid range for feature $f [$a]\n";
             }
	  }
       }
       my @feature;
       if ($action[0] eq "C") { #MULTIPLE-COLUMNS
          @feature = ($action[0], $column, \@args);
       }
       else {
          @feature = ($action[0], \@args);
       }
       push(@fs, \@feature);
    }

    return(\@fs);
}

sub read_fs
{
    #description _ conveniently reads and transforms a feature set list file
    #param1 _ filename
    #param2 _ window setup
    #@return _ feature set list

    my $file = shift;
    my $winsetup = shift;

    my $FILE = new IO::File("< $file") or die "Couldn't open input file: $file\n";
    my @list;

    my $minpos = 0 - $winsetup->[1];
    my $maxpos = $winsetup->[0] - $winsetup->[1] - 1;

    while (defined(my $line = $FILE->getline())) {
       chomp($line);
       my @action = split(/[()]/, $line);
       my @args;
       my $column;
       if ($action[0] eq "C") { #MULTIPLE-COLUMNS
          my @ARGS = split(/[;]/, $action[1]);
          $column = $ARGS[0];
          @args = split(/[,]/, $ARGS[1]);
       }
       else {
          @args = split(/[,]/, $action[1]);
       }
       my $valid = 1;
       if (ATTGEN::check_arguments($action[0])) {
          if (scalar(@args) == 0) { die "[WRONG FEATURE!!] a range must be specified for feature $action[0]\n"; }
          foreach my $a (@args) {
             if (!(($a >= $minpos) and ($a <= $maxpos))) {
                die "[WRONG FEATURE!!] invalid range for feature $action[0] [$a]\n";
             }
	  }
       }
       my @feature;
       if ($action[0] eq "C") { #MULTIPLE-COLUMNS
          @feature = ($action[0], $column, \@args);
       }
       else {
          @feature = ($action[0], \@args);
       }
       push(@list, \@feature);
    }

    $FILE->close();

    return(\@list);
}

sub read_config_file
{
    #description _ reads an SVMT configuration file and creates
    #              the corresponiding config hash.
    #param1  _ config-filename
    #@return _ config hash reference

    my $config = shift;

    my %CONFIG;
    my $fCONFIG = new IO::File("< $config") or die "Couldn't open configuration file: $config\n";

    $CONFIG{wlen} = $SVMTAGGER::wlength;
    $CONFIG{wcore} = $SVMTAGGER::wcorepos;
    $CONFIG{F} = 0;
    $CONFIG{Ck} = $SVMTAGGER::Ck;
    $CONFIG{Cu} = $SVMTAGGER::Cu;
    $CONFIG{X} = $SVMTAGGER::X;
    $CONFIG{Dratio} = $SVMTAGGER::dicterr;
    $CONFIG{Eratio} = $SVMTAGGER::eqerr;
    $CONFIG{Kfilter} = $SVMTAGGER::Kfilter;
    $CONFIG{Ufilter} = $SVMTAGGER::Ufilter;
    $CONFIG{R} = "";
    $CONFIG{BLEX} = "";
    $CONFIG{LEX} = "";
    $CONFIG{A0k} = $SVMTAGGER::rATT0k;
    $CONFIG{A1k} = $SVMTAGGER::rATT1k;
    $CONFIG{A2k} = $SVMTAGGER::rATT2k;
    $CONFIG{A3k} = $SVMTAGGER::rATT3k;
    $CONFIG{A4k} = $SVMTAGGER::rATT0k;
    $CONFIG{A0u} = $SVMTAGGER::rATT0u;
    $CONFIG{A1u} = $SVMTAGGER::rATT1u;
    $CONFIG{A2u} = $SVMTAGGER::rATT2u;
    $CONFIG{A3u} = $SVMTAGGER::rATT3u;
    $CONFIG{A4u} = $SVMTAGGER::rATT0u;
    $CONFIG{rmfiles} = $SVMTAGGER::rmfiles;
    $CONFIG{remakeFLDS} = $SVMTAGGER::remakeFLDS;
    $CONFIG{TRAINP} = $SVMTAGGER::TRAINP;
    $CONFIG{VALP} = $SVMTAGGER::VALP;
    $CONFIG{TESTP} = $SVMTAGGER::TESTP;

    my @DO;

    my $l = 1;
    while (defined(my $line = $fCONFIG->getline())) {
       chomp($line);
       if (($line ne "") and ($line !~ /^\#.*/)) {
	  $line =~ s/( )*=( )*/=/g;
          if ($line =~ /^do .*/) {
	      if ($line =~ /^[^=]*=[^=]*/) { print STDERR "ERRONEOUS CONFIG-FILE: $config [line $l]\n"; exit; }
              my @command = split(/ +/, $line);
              if (scalar(@command) < 3) { print STDERR "ERRONEOUS CONFIG-FILE (Wrong DO Action) [line $l]: $config\n"; exit; }
              if ($command[0] eq "do") {
		 my $extra;
		 if (scalar(@command) > 3) { $command[3] =~ s/:/;/g; $extra .= ":".$command[3]; }
		 if (scalar(@command) > 4) { $command[4] =~ s/:/;/g; $extra .= ":".$command[4]; }
		 if (scalar(@command) > 5) { $command[5] =~ s/:/;/g; $extra .= ":".$command[5]; }
		 if ($command[1] eq "M0") {
		     if ($command[2] eq "LR"){ push(@DO, "0:LR$extra"); }
		     if ($command[2] eq "RL"){ push(@DO, "0:RL$extra"); }
		     if ($command[2] eq "LRL"){ push(@DO, "0:LR$extra"); push(@DO, "0:RL$extra"); }
		 }
		 if ($command[1] eq "M1") {
		     if ($command[2] eq "LR"){ push(@DO, "1:LR$extra"); }
		     if ($command[2] eq "RL"){ push(@DO, "1:RL$extra"); }
		     if ($command[2] eq "LRL"){ push(@DO, "1:LR$extra"); push(@DO, "1:RL$extra"); }
		 }
		 if ($command[1] eq "M2") {
		     if ($command[2] eq "LR"){ push(@DO, "2:LR$extra"); }
		     if ($command[2] eq "RL"){ push(@DO, "2:RL$extra"); }
		     if ($command[2] eq "LRL"){ push(@DO, "2:LR$extra"); push(@DO, "2:RL$extra"); }
		 }
		 if ($command[1] eq "M3") {
		     if ($command[2] eq "LR"){ push(@DO, "3:LR$extra"); }
		     if ($command[2] eq "RL"){ push(@DO, "3:RL$extra"); }
		     if ($command[2] eq "LRL"){ push(@DO, "3:LR$extra"); push(@DO, "3:RL$extra"); }
		 }
		 if ($command[1] eq "M4") {
		     if ($command[2] eq "LR"){ push(@DO, "4:LR$extra"); }
		     if ($command[2] eq "RL"){ push(@DO, "4:RL$extra"); }
		     if ($command[2] eq "LRL"){ push(@DO, "4:LR$extra"); push(@DO, "4:RL$extra"); }
		 }
	      }
	  }
          else {
	      if ($line !~ /^[^=]*=[^=]*/) { print STDERR "ERRONEOUS CONFIG-FILE: $config [line $l]\n"; exit; }
	      my @entry = split(/=/, $line);
              my $opt = $entry[0];
              my @args = split(/ +/, $entry[1]);
              if (scalar(@args) == 0) { print STDERR "ERRONEOUS CONFIG-FILE: $config [line $l]\n"; exit; }
              if ($opt eq "NAME") {
		  $CONFIG{model} = $args[0];
	      }
              elsif ($opt eq "SET") {
		  if (-e $args[0]) { $CONFIG{set} = $args[0]; }
	      }
              elsif ($opt eq "TRAINP") { $CONFIG{TRAINP} = $args[0]; }
              elsif ($opt eq "VALP") { $CONFIG{VALP} = $args[0]; }
              elsif ($opt eq "TESTP") { $CONFIG{TESTP} = $args[0]; }
              elsif ($opt eq "TRAINSET") {
		  if (-e $args[0]) { $CONFIG{trainset} = $args[0]; }
	      }
              elsif ($opt eq "VALSET") {
		  if (-e $args[0]) { $CONFIG{valset} = $args[0]; }
	      }
              elsif ($opt eq "TESTSET") {
		  if (-e $args[0]) { $CONFIG{testset} = $args[0]; }
	      }
              elsif ($opt eq "SVMDIR") {
		  if (-e $args[0]) { $CONFIG{SVMDIR} = $args[0]; }
	      }
              elsif ($opt eq "R") {
		  if (-e $args[0]) { $CONFIG{R} = $args[0]; }
	      }
              elsif ($opt eq "BLEX") {
		  if (-e $args[0]) { $CONFIG{BLEX} = $args[0]; }
	      }
              elsif ($opt eq "LEX") {
		  if (-e $args[0]) { $CONFIG{LEX} = $args[0]; }
	      }
              elsif ($opt eq "W") {
                  if (scalar(@args) < 2) { print STDERR "ERRONEOUS CONFIG-FILE (Wrong Sliding Window Definition) [line $l]: $config\n"; exit; }
                  if ($args[0] < $args[1]) { print STDERR "ERRONEOUS CONFIG-FILE (Wrong Sliding Window Definition) [line $l]: $config\n"; exit; }
		  $CONFIG{wlen} = $args[0];
		  $CONFIG{wcore} = $args[1];
	      }
              elsif ($opt eq "F") {
                  if (scalar(@args) < 2) { print STDERR "ERRONEOUS CONFIG-FILE (Wrong Feature Filtering Definition) [line $l]: $config\n"; exit; }
                  $CONFIG{F} = 1;
		  $CONFIG{fmin} = $args[0];
		  $CONFIG{maxmapsize} = $args[1];
	      }
              elsif ($opt eq "CK") { $CONFIG{Ck} = $args[0]; }
              elsif ($opt eq "CU") { $CONFIG{Cu} = $args[0]; }
              elsif ($opt eq "X") { $CONFIG{X} = $args[0]; }
              elsif ($opt eq "Dratio") { $CONFIG{Dratio} = $args[0]; }
              elsif ($opt eq "Eratio") { $CONFIG{Eratio} = $args[0]; }
              elsif ($opt eq "Kfilter") { $CONFIG{Kfilter} = $args[0]; }
              elsif ($opt eq "Ufilter") { $CONFIG{Ufilter} = $args[0]; }
              elsif (($opt eq "REMOVE_FILES") or ($opt eq "RM")) { $CONFIG{rmfiles} = $args[0]; }
              elsif (($opt eq "REMAKE_FOLDERS") or ($opt eq "RF")) { $CONFIG{remakeFLDS} = $args[0]; }
              elsif ($opt eq "AP") {
                 if (scalar(@args) > 0) { $CONFIG{AP} = \@args; }
              }
              elsif ($opt eq "UP") {
		 if (scalar(@args) > 0) { $CONFIG{UP} = \@args; }
              }
 	      elsif ($opt eq "A0k") {
                 if (scalar(@args) > 0) { $CONFIG{A0k} = \@args; }
              }
              elsif ($opt eq "A1k") {
                 if (scalar(@args) > 0) { $CONFIG{A1k} = \@args; }
              }
  	      elsif ($opt eq "A2k") {
                 if (scalar(@args) > 0) { $CONFIG{A2k} = \@args; }
              }
  	      elsif ($opt eq "A3k") {
                 if (scalar(@args) > 0) { $CONFIG{A3k} = \@args; }
              }
  	      elsif ($opt eq "A4k") {
                 if (scalar(@args) > 0) { $CONFIG{A4k} = \@args; }
              }
 	      elsif ($opt eq "A0u") {
                 if (scalar(@args) > 0) { $CONFIG{A0u} = \@args; }
              }
              elsif ($opt eq "A1u") {
                 if (scalar(@args) > 0) { $CONFIG{A1u} = \@args; }
              }
  	      elsif ($opt eq "A2u") {
                 if (scalar(@args) > 0) { $CONFIG{A2u} = \@args; }
              }
  	      elsif ($opt eq "A3u") {
                 if (scalar(@args) > 0) { $CONFIG{A3u} = \@args; }
              }
  	      elsif ($opt eq "A4u") {
                 if (scalar(@args) > 0) { $CONFIG{A4u} = \@args; }
              }
          }
       }
       $l++;
    }
    if ((!(exists($CONFIG{set}))) and (!(exists($CONFIG{trainset})))) { print STDERR "ERRONEOUS CONFIG-FILE [BAD/MISSING TRAINSET]: $config\n"; exit; }
    if (!(exists($CONFIG{model}))) { print STDERR "ERRONEOUS CONFIG-FILE [BAD/MISSING NAME]: $config\n"; exit; }
    if (!(exists($CONFIG{SVMDIR}))) { print STDERR "ERRONEOUS CONFIG-FILE [BAD/MISSING SVMDIR]: $config\n"; exit; }

    $CONFIG{do} = \@DO;

    $fCONFIG->close();
    return \%CONFIG;
}

# --------------- RANDOMIZE ----------------------------------------------------------------

sub randomize_sentences{
    #description _ Randomly distributes a corpus sentences into 3 sets...
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
    print "<$trainset> - ", $pctrain * 100, "%\n";
    print "<$valset> - ", $pcval * 100, "%\n";
    print "<$testset> - ", $pctest * 100, "%\n";

    srand();
    my $rn;

    my $ntrain = 0;
    my $nval = 0;
    my $ntest = 0;
    my $wtrain = 0;
    my $wval = 0;
    my $wtest = 0;

    my $sentence = "";
    my $nwords = 0;
    while (defined(my $line = $CORPUS->getline())) {
        $nwords++;
        $sentence .= $line;
        chomp($line);
        my @item = split($COMMON::in_valseparator, $line);
        if (COMMON::end_of_sentence($item[0])) {
           $rn = rand($Xtrain + $Xval + $Xtest);
           if ($rn < $Xtrain) { print $TRAIN $sentence; $ntrain++; $wtrain += $nwords; }
           elsif (($rn >= $Xtrain) && ($rn < $Xtrain + $Xval)) { print $VAL $sentence; $nval++; $wval += $nwords; } 
	   elsif (($rn >= $Xtrain + $Xval) && ($rn <= $Xtrain + $Xval + $Xtest)) { print $TEST $sentence; $ntest++;  $wtest += $nwords;}
           $sentence = "";
           $nwords = 0;
        }
    }

    print "TRAINING   -> $ntrain sentences :: $wtrain words\n";
    print "VALIDATION -> $nval sentences :: $wval words\n";
    print "TEST       -> $ntest sentences :: $wtest words\n";

    $TEST->close();
    $VAL->close();
    $TRAIN->close();
    $CORPUS->close();
}

# --------------- DICTIONARY ---------------------------------------------------------------

sub create_dictionary
{
   #description _ dictionary creation [construction, expansion, repairing]
   #param1  _ source corpus
   #param2  _ source dictionary (for unsupervised learning)
   #param3  _ backup lexicon
   #param4  _ repairing list
   #param5  _ dictionary error ratio for automatic repairing
   #param6  _ target dictionary filename
   #param7  _ report filename
   #param8  _ verbosity
   #@return _ supervised corpus?

   my $trainset = shift;
   my $lex = shift;
   my $blex = shift;
   my $RLIST = shift;
   my $Dratio = shift;
   my $dict = shift;   
   my $reportfile = shift;
   my $verbose = shift;

   if ($verbose > $COMMON::verbose1) { COMMON::report($reportfile, "CREATING DICTIONARY <$dict> FROM <$trainset>\n"); }
   my $supervised = SVMTAGGER::generate_dictionary($trainset, $dict, 0, -1);
   if (!$supervised) {
      if ($lex ne "") {
         if ($verbose > $COMMON::verbose1) { COMMON::report($reportfile, "LEXICON... <$lex>\n"); }
         system "cp $lex $dict";
      }
      else { die "[LEX] UNSUPERVISED LEXICON UNDEFINED\n"; }
   }
   srand();
   my $DICT = $dict.".".rand(100000);
   if (($blex ne "") and (-e $blex)) {
      SVMTAGGER::expand_dictionary($dict, $blex, $DICT);
      if ($verbose > $COMMON::verbose1) { COMMON::report($reportfile, "EXPANDING DICTIONARY <$dict> [using ".$blex."]\n"); }
      system ("mv $DICT $dict");
   }
   if (($RLIST ne "") and (-e $RLIST)) {
      my $N = SVMTAGGER::repair_dictionary($dict, $RLIST, $DICT);
      if ($verbose > $COMMON::verbose1) { COMMON::report($reportfile, "REPAIRING DICTIONARY <$dict> [using ".$RLIST."] (#REPAIRINGS = $N)\n"); }
      system ("mv $DICT $dict");
   }

   my $N = SVMTAGGER::repair_dictionary_H($dict, $DICT, $Dratio);
   if ($verbose > $COMMON::verbose1) { COMMON::report($reportfile, "AUTOMATICALLY REPAIRING DICTIONARY <$dict> [$Dratio] (#REPAIRINGS = $N)\n"); }
   system ("mv $DICT $dict");

   return $supervised;
}

sub generate_dictionary
{
   #description _ generates a dictionary out from a given corpus
   #              skipping the portion from [$start..$end] both
   #              $start and $end skipped
   #param1  _ TRAINING CORPUS filename
   #param2  _ TARGET DICTIONARY filename
   #param3  _ slice START POINT   (if START ==  0  then read from the beginning of the corpus)
   #param4  _ slice END POINT     (if   END == -1  then read until the end of the corpus)
   #@return _ supervised corpus?

   my $corpus = shift;
   my $dict = shift;
   my $start = shift;
   my $end = shift;

   #Dictionary format : <word> <n_occurrences> <n_POS> {(POS1, n_oc1)..(POSN, n_ocN)}

   my $CORPUS = new IO::File("< $corpus") or die "Couldn't open input file: $corpus\n";
   my $supervised = 1;

   my $i = 0;
   my %DICT;
   while (defined(my $line = $CORPUS->getline())) {
      chomp($line);
      if ($line ne "") {
         if (($i < $start) or ($i > $end)) {
            my @entry = split($COMMON::in_valseparator, $line);
            if (scalar(@entry) > 0) { #line is not empty
               if (scalar(@entry) == 1) {  	       
                  if (exists($DICT{$entry[0]})) {
                     $DICT{$entry[0]}[0]++;
   	          }
                  else {
                     my %l; 
                      my @v = (1, \%l);
  	             $DICT{$entry[0]} = \@v;               
	          }
                  $supervised = 0;
               }
               else {
                  #%DICT <word> --> [n_occurrences] [%l --> {(POS1, n_oc1)..(POSN, n_ocN)}]
                  if (exists($DICT{$entry[0]})) {
                     $DICT{$entry[0]}[0]++;
	             my $l = $DICT{$entry[0]}[1];
                     if (exists($l->{$entry[1]})) {  $l->{$entry[1]}++; }
                     else { $l->{$entry[1]} = 1; }
                  }
                  else { 
                     my %l = ("$entry[1]" => 1); 
                     my @v = (1, \%l);
	             $DICT{$entry[0]} = \@v;
                  }
	       }
	    }
	 }
      }
      $i++;
   }

   $CORPUS->close();

   my $DICTIONARY = new IO::File("> $dict") or die "Couldn't open output file: $dict\n";
   #Dictionary format : <word> <n_occurrences> <n_POS> {(POS1, n_oc1)..(POSN, n_ocN)}
   foreach my $w (sort keys %DICT) {
       my $entry = $w.$COMMON::d_separator.$DICT{$w}[0].$COMMON::d_separator.scalar(keys %{$DICT{$w}[1]});
       foreach my $p (sort keys %{$DICT{$w}[1]}) { #sorting is important for ambiguity classes
          $entry .= $COMMON::d_separator.$p.$COMMON::d_separator.$DICT{$w}[1]->{$p};
       }
       $entry =~ s/ +/ /g;
       print $DICTIONARY "$entry\n";
   }

   $DICTIONARY->close();

   return $supervised;
}

sub repair_dictionary
{
   #description _ wisely repairs a SVMT dictionary file
   #param1 _ dictionary to repair filename
   #param2 _ 200 repaired word list
   #param3 _ TARGET DICTIONARY filename
   #@return _ number of repairings

   my $dict = shift;
   my $WSJ200 = shift;
   my $DICT = shift;

   my $N = 0;

   my $LEX = new IO::File("< $dict") or die "Couldn't open input file: $dict\n";
   my $LIST = new IO::File("< $WSJ200") or die "Couldn't open input file: $WSJ200\n"; 
   my $OUT = new IO::File("> $DICT") or die "Couldn't open output file: $DICT\n";

   my %NEW;
   while (defined(my $elem = $LIST->getline())) {
       chomp($elem);
       my @l = split($COMMON::d_separator, $elem);
       my $key = $l[0];
       shift(@l);
       shift(@l);
       shift(@l);
       my $i = 0;
       my %h;
       while ($i < scalar(@l)) {
	   $h{$l[$i]} = 1;
           $i+=2;
       }
       $NEW{$key} = \%h;
   }

   my %OUTH;
   my %CARD;
   my %CARDPUNCT;
   my %CARDSUFFIX;
   my %CARDSEPS;
   $CARD{"n_occ"} = 0;
   $CARDPUNCT{"n_occ"} = 0;
   $CARDSUFFIX{"n_occ"} = 0;
   $CARDSEPS{"n_occ"} = 0;

   while (defined(my $entry = $LEX->getline())) {
       chomp($entry);
       my @line = split($COMMON::d_separator, $entry);
       my $word = $line[0];
       my $nocc = $line[1];
       my $npos = $line[2];
       if ($word =~ /^[0-9]+$/) {
          $CARD{"n_occ"} += $nocc;
          my $i = 3;
          while ($i < scalar(@line)) {
             if (exists($CARD{$line[$i]})) { $CARD{$line[$i]} += $line[$i+1]; }
             else { $CARD{$line[$i]} = $line[$i+1]; }
             $i+=2
          }
       }
       elsif ($word =~ /^[0-9]+[\.\,\!\?:]+$/) {
          $CARDPUNCT{"n_occ"} += $nocc;
          my $i = 3;
          while ($i < scalar(@line)) {
             if (exists($CARDPUNCT{$line[$i]})) { $CARDPUNCT{$line[$i]} += $line[$i+1]; }
             else { $CARDPUNCT{$line[$i]} = $line[$i+1]; }
             $i+=2
          }
       }
       elsif ($word =~ /^[0-9]+[:\.\,\/\\\-][0-9\.\,\-\\\/]+$/) {
          $CARDSEPS{"n_occ"} += $nocc;
          my $i = 3;
          while ($i < scalar(@line)) {
             if (exists($CARDSEPS{$line[$i]})) { $CARDSEPS{$line[$i]} += $line[$i+1]; }
             else { $CARDSEPS{$line[$i]} = $line[$i+1]; }
             $i+=2
          }
       }
       elsif ($word =~ /^[0-9]+[^0-9]+.*$/) {
          $CARDSUFFIX{"n_occ"} += $nocc;
          my $i = 3;
          while ($i < scalar(@line)) {
             if (exists($CARDSUFFIX{$line[$i]})) { $CARDSUFFIX{$line[$i]} += $line[$i+1]; }
             else { $CARDSUFFIX{$line[$i]} = $line[$i+1]; }
             $i+=2
          }
       }

       if (exists($NEW{$word})) { # entry must be repaired
  	     my $newline;
             shift(@line);
             shift(@line);
             shift(@line);
             my $i = 0;
             my $newnpos = 0;
             while ($i < scalar(@line)) {
	        if (exists($NEW{$word}->{$line[$i]})) { $newline .=  $COMMON::d_separator.$line[$i].$COMMON::d_separator.$line[$i+1]; $newnpos++; }
                else { $N++; }
                $i+=2;
             }
             $OUTH{$word} = $word.$COMMON::d_separator.$nocc.$COMMON::d_separator.$newnpos.$newline;
       }
       else { # entry stays the same
             $OUTH{$word} = $entry;
       }
   }

   # ------------------------------------------------------------------
   my $len = scalar(keys(%CARD)) - 1;
   my $card = "\@CARD ".$CARD{"n_occ"}.$COMMON::d_separator.$len.$COMMON::d_separator;
   foreach my $c (sort keys %CARD) { 
      if ($c ne "n_occ") { $card .= $c.$COMMON::d_separator.$CARD{$c}.$COMMON::d_separator; }
   }
   chop($card);
   $OUTH{"\@CARD"} = $card;
   # ------------------------------------------------------------------
   $len = scalar(keys(%CARDPUNCT)) - 1;
   $card = "\@CARDPUNCT ".$CARDPUNCT{"n_occ"}.$COMMON::d_separator.$len.$COMMON::d_separator;
   foreach my $c (sort keys %CARDPUNCT) { 
      if ($c ne "n_occ") { $card .= $c.$COMMON::d_separator.$CARDPUNCT{$c}.$COMMON::d_separator; }
   }
   chop($card);
   $OUTH{"\@CARDPUNCT"} = $card;
   # ------------------------------------------------------------------
   $len = scalar(keys(%CARDSUFFIX)) - 1;
   $card = "\@CARDSUFFIX ".$CARDSUFFIX{"n_occ"}.$COMMON::d_separator.$len.$COMMON::d_separator;
   foreach my $c (sort keys %CARDSUFFIX) { 
      if ($c ne "n_occ") { $card .= $c.$COMMON::d_separator.$CARDSUFFIX{$c}.$COMMON::d_separator; }
   }
   chop($card);
   $OUTH{"\@CARDSUFFIX"} = $card;
   # ------------------------------------------------------------------
   $len = scalar(keys(%CARDSEPS)) - 1;
   $card = "\@CARDSEPS ".$CARDSEPS{"n_occ"}.$COMMON::d_separator.$len.$COMMON::d_separator;
   foreach my $c (sort keys %CARDSEPS) { 
      if ($c ne "n_occ") { $card .= $c.$COMMON::d_separator.$CARDSEPS{$c}.$COMMON::d_separator; }
   }
   chop($card);
   $OUTH{"\@CARDSEPS"} = $card;
   # ------------------------------------------------------------------

   foreach my $v (sort keys %OUTH) {
      my $entry .= $OUTH{$v};
      $entry =~ s/ +/ /g;
      print $OUT "$entry\n";
   }

   # ------------------------------------------------------------------

   $LEX->close();
   $LIST->close();
   $OUT->close();

   return($N);
}

sub repair_dictionary_H
{
   #description _ wisely repairs a SVMT dictionary file
   #param1 _ dictionary to repair filename
   #param2 _ TARGET DICTIONARY filename
   #param3 _ relevance coefficient
   #return _ number of repairings

   my $dict = shift;
   my $DICT = shift;
   my $CR = shift;

   my $N = 0;

   my $LEX = new IO::File("< $dict") or die "Couldn't open input file: $dict\n";
   my $OUT = new IO::File("> $DICT") or die "Couldn't open output file: $DICT\n";

   my %OUTH;
   my %CARD;
   my %CARDPUNCT;
   my %CARDSUFFIX;
   my %CARDSEPS;
   $CARD{"n_occ"} = 0;
   $CARDPUNCT{"n_occ"} = 0;
   $CARDSUFFIX{"n_occ"} = 0;
   $CARDSEPS{"n_occ"} = 0;

   while (defined(my $entry = $LEX->getline())) {
       chomp($entry);
       my @line = split($COMMON::d_separator, $entry);
       my $word = $line[0];
       my $nocc = $line[1];
       my $npos = $line[2];
       if ($word =~ /^[0-9]+$/) {
          $CARD{"n_occ"} += $nocc;
          my $i = 3;
          while ($i < scalar(@line)) {
             if (exists($CARD{$line[$i]})) { $CARD{$line[$i]} += $line[$i+1]; }
             else { $CARD{$line[$i]} = $line[$i+1]; }
             $i+=2
          }
       }
       elsif ($word =~ /^[0-9]+[\.\,\!\?:]+$/) {
          $CARDPUNCT{"n_occ"} += $nocc;
          my $i = 3;
          while ($i < scalar(@line)) {
             if (exists($CARDPUNCT{$line[$i]})) { $CARDPUNCT{$line[$i]} += $line[$i+1]; }
             else { $CARDPUNCT{$line[$i]} = $line[$i+1]; }
             $i+=2
          }
       }
       elsif ($word =~ /^[0-9]+[:\.\,\/\\\-][0-9\.\,\-\\\/]+$/) {
          $CARDSEPS{"n_occ"} += $nocc;
          my $i = 3;
          while ($i < scalar(@line)) {
             if (exists($CARDSEPS{$line[$i]})) { $CARDSEPS{$line[$i]} += $line[$i+1]; }
             else { $CARDSEPS{$line[$i]} = $line[$i+1]; }
             $i+=2
          }
       }
       elsif ($word =~ /^[0-9]+[^0-9]+.*$/) {
          $CARDSUFFIX{"n_occ"} += $nocc;
          my $i = 3;
          while ($i < scalar(@line)) {
             if (exists($CARDSUFFIX{$line[$i]})) { $CARDSUFFIX{$line[$i]} += $line[$i+1]; }
             else { $CARDSUFFIX{$line[$i]} = $line[$i+1]; }
             $i+=2
          }
       }

       my $newline;
       my $neww = shift(@line);
       my $newn = shift(@line);
       my $newp = shift(@line);
       my $i = 0;
       while ($i < scalar(@line)) {
	  if ($line[$i+1] < ($newn * $CR)) {
	     $N++;
             $newp--;
             $newn -= $line[$i+1];
          }
          else {
             $newline .= $COMMON::d_separator.$line[$i].$COMMON::d_separator.$line[$i+1];
          }
          $i+=2;
       }
       $OUTH{$word} = "$neww $newn $newp$newline";

   }

   # ------------------------------------------------------------------
   my $len = scalar(keys(%CARD)) - 1;
   my $card = "\@CARD ".$CARD{"n_occ"}.$COMMON::d_separator.$len.$COMMON::d_separator;
   foreach my $c (sort keys %CARD) { 
      if ($c ne "n_occ") { $card .= $c.$COMMON::d_separator.$CARD{$c}.$COMMON::d_separator; }
   }
   chop($card);
   $OUTH{"\@CARD"} = $card;
   # ------------------------------------------------------------------
   $len = scalar(keys(%CARDPUNCT)) - 1;
   $card = "\@CARDPUNCT ".$CARDPUNCT{"n_occ"}.$COMMON::d_separator.$len.$COMMON::d_separator;
   foreach my $c (sort keys %CARDPUNCT) { 
      if ($c ne "n_occ") { $card .= $c.$COMMON::d_separator.$CARDPUNCT{$c}.$COMMON::d_separator; }
   }
   chop($card);
   $OUTH{"\@CARDPUNCT"} = $card;
   # ------------------------------------------------------------------
   $len = scalar(keys(%CARDSUFFIX)) - 1;
   $card = "\@CARDSUFFIX ".$CARDSUFFIX{"n_occ"}.$COMMON::d_separator.$len.$COMMON::d_separator;
   foreach my $c (sort keys %CARDSUFFIX) { 
      if ($c ne "n_occ") { $card .= $c.$COMMON::d_separator.$CARDSUFFIX{$c}.$COMMON::d_separator; }
   }
   chop($card);
   $OUTH{"\@CARDSUFFIX"} = $card;
   # ------------------------------------------------------------------
   $len = scalar(keys(%CARDSEPS)) - 1;
   $card = "\@CARDSEPS ".$CARDSEPS{"n_occ"}.$COMMON::d_separator.$len.$COMMON::d_separator;
   foreach my $c (sort keys %CARDSEPS) { 
      if ($c ne "n_occ") { $card .= $c.$COMMON::d_separator.$CARDSEPS{$c}.$COMMON::d_separator; }
   }
   chop($card);
   $OUTH{"\@CARDSEPS"} = $card;
   # ------------------------------------------------------------------

   foreach my $v (sort keys %OUTH) {
      my $entry .= $OUTH{$v};
      $entry =~ s/ +/ /g;
      print $OUT "$entry\n";
   }

   # ------------------------------------------------------------------

   $LEX->close();
   $OUT->close();

   return($N);
}

sub enrich_dictionary
{
   #description _ a dictionary is enriched
   #             (entries matching an expanded dictionary are enriched)
   #param1 _ dictionary to enrich (filename)
   #param2 _ expanded dictionary  (filename)

   my $dict = shift;
   my $dictexpnd = shift;

   my $DICT = new IO::File("< $dict") or die "Couldn't open input file: $dict\n";
   my $DICTEXPND = new IO::File("< $dictexpnd") or die "Couldn't open input file: $dictexpnd\n";

   #%DICT <word> --> [n_occurrences] [n_pos --> {(POS1, n_oc1)..(POSN, n_ocN)}]
   my %D;
   while (defined(my $line = $DICT->getline())) {
      chomp($line);
      my @entry = split($COMMON::d_separator, $line);
      my %posl;
      my $i = 0;
      while ($i < 2 * $entry[2]) {
         $posl{$entry[$i+3]} = $entry[$i+4];
         $i += 2;
      }
      my @l = ($entry[1], \%posl);
      $D{$entry[0]} = \@l;
   }
   $DICT->close();

   while (defined(my $line = $DICTEXPND->getline())) {
      chomp($line);
      my @entry = split($COMMON::d_separator, $line);
      if (exists($D{$entry[0]})) {
         my $i = 0;
         while ($i < 2 * $entry[2]) {
            if (!exists($D{$entry[0]}->[1]->{$entry[$i+3]})) {
	       $D{$entry[0]}->[1]->{$entry[$i+3]} = $entry[$i+4];
	    }
            $i += 2;
         }
      }
   }
   $DICTEXPND->close();

   my $DICT = new IO::File("> $dict") or die "Couldn't open output file: $dict\n";
   foreach my $w (sort keys %D) {
       my $entry = $w.$COMMON::d_separator.$D{$w}[0].$COMMON::d_separator.scalar(keys %{$D{$w}[1]});
       foreach my $p (sort keys %{$D{$w}[1]}) {
          $entry .= $COMMON::d_separator.$p.$COMMON::d_separator.$D{$w}[1]->{$p};
       }
       $entry =~ s/ +/ /g;
       print $DICT "$entry\n";
   }
   $DICT->close();
   
}

sub expand_dictionary
{
   #description _ merges two different SVMT dictionary files onto a single one
   #              (entries matching and not matching)
   #param1 _ dictionary to expand filename
   #param2 _ backup lexicon
   #param3 _ TARGET DICTIONARY filename
   #param4 _ VERBOSE

   my $dict = shift;
   my $blex = shift;
   my $DICT = shift;
   my $verbose = shift;

   my $LEX = new IO::File("< $dict") or die "Couldn't open input file: $dict\n";
   my $BLEX = new IO::File("< $blex") or die "Couldn't open input file: $blex\n"; 
   my $OUT = new IO::File("> $DICT") or die "Couldn't open output file: $DICT\n";


   if ($verbose) { print STDERR "READING DICTIONARY <$dict>\n"; }
   my %OUTH;
   my $iter = 0;
   while (defined(my $entry = $LEX->getline())) {
       chomp($entry);
       my @line = split($COMMON::d_separator, $entry);
       my $word = shift(@line);
       my $nocc = shift(@line);
       my $npos = shift(@line);
       my $i = 0;
       my %h;
       while ($i < scalar(@line)) { #reading dictionary entry
          $h{$line[$i]} = $line[$i+1];
          $i+=2;
       }
       my @laux = ($nocc, $npos, \%h);
       $OUTH{$word} = \@laux;
       $iter++;
       if ($verbose) { COMMON::show_progress($iter, $COMMON::progress1, $COMMON::progress2); }
   }
   $LEX->close();

   if ($verbose) { print STDERR ".$iter [DONE]\nREADING BACKUP LEXICON <$blex>\n"; }
   my %NEW;
   $iter = 0;
   while (defined(my $elem = $BLEX->getline())) {
       chomp($elem);
       my @line = split($COMMON::d_separator, $elem);
       my $word = shift(@line);
       my $nocc = shift(@line);
       my $npos = shift(@line);
       if (exists($OUTH{$word})) { #entry may be expanded / may not
          my $i = 0;
          while ($i < scalar(@line)) {
	     if (!(exists($OUTH{$word}->[2]->{$line[$i]}))) {
                $OUTH{$word}->[1]++;
		$OUTH{$word}->[2]->{$line[$i]} = 0;
	     }
             $OUTH{$word}->[2]->{$line[$i]} += $line[$i+1];
             $OUTH{$word}->[0] += $line[$i+1];
             $i+=2;
          }
       }
       else { #new entry is appended
          $OUTH{$word} = $elem;
          my $i = 0;
          my %h;
          while ($i < scalar(@line)) { #reading dictionary entry
             $h{$line[$i]} = $line[$i+1];
             $i+=2;
          }
          my @laux = ($nocc, $npos, \%h);
          $OUTH{$word} = \@laux;
       }
       $iter++;
       if ($verbose) { COMMON::show_progress($iter, $COMMON::progress1, $COMMON::progress2); }
   }
   $BLEX->close();

   if ($verbose) { print STDERR ".$iter [DONE]\nWRITING EXPANDED DICTIONARY <$DICT>\n"; }
   $iter = 0;
   foreach my $v (sort keys %OUTH) {
      my $entry = $v.$COMMON::d_separator.$OUTH{$v}[0].$COMMON::d_separator.$OUTH{$v}[1];
      foreach my $p (sort keys %{$OUTH{$v}[2]}) {
         $entry .= $COMMON::d_separator.$p.$COMMON::d_separator.$OUTH{$v}[2]->{$p};
      }
      $entry =~ s/ +/ /g;
      print $OUT "$entry\n";
      $iter++;
      if ($verbose) { COMMON::show_progress($iter, $COMMON::progress1, $COMMON::progress2); }
   }
   $OUT->close();

   if ($verbose) { print STDERR ".$iter [DONE]\n"; }
}

# --------------- AUXILIAR -----------------------------------------------------------------
 
sub any_unknown
{
    #description _ returns true only if  true if there's any unknown word at the right of the
    #              core previous to the end/beginning of the sentence.
    #param1  _ window containing the candidate to be ambiguous word as its core item
    #param2  _ dictionary object reference
    #@return _ true if there's any unknown word at the right of the core previous
    #          to the end/beginning of the sentence.

    my $rwin = shift;
    my $rdict = shift;

    my $i = $rwin->get_core;
    my $eos = 0;
    my $unknown = 0;
    while (($i < $rwin->get_len) and (!$eos) and (!$unknown)) {
       my $w = $rwin->get_word($i);
       if (COMMON::end_of_sentence($w) or ($w eq $COMMON::emptyword)) { $eos = 1; }
       else { $unknown = $rdict->unknown_word($w); }
       $i++;
    }

    return $unknown;
}
 
sub frequent_word_window
{
    #description _ returns true only if the window core word is reasonably frequent
    #param1 _ window containing the candidate to be ambiguous word as its core item
    #param2 _ dictionary object reference

    my $rwin = shift;
    my $rdict = shift;

    return $rdict->frequent_word($rwin->get_core_word());
}
 
sub ambiguous_word_window
{
    #description _ returns true only if the window core word is a POS ambiguous one
    #param1 _ window containing the candidate to be ambiguous word as its core item
    #param2 _ dictionary object reference
    #param3 _ mode (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #              (                  3-unsupervised :: 4-unkown words on training)    

    my $rwin = shift;
    my $rdict = shift;
    my $mode = shift;

    if ($mode == $COMMON::mode3) { return $rdict->ambiguous_word($rwin->get_core_actual_word()); }
    else { return $rdict->ambiguous_word($rwin->get_core_word()); }

    #return $rdict->ambiguous_word($rwin->get_core_word());
}

sub unknown_word_window
{
    #description _ returns true only if the window core word is an unknown one
    #param1 _ window containing the candidate to be unknown word as its core item
    #param2 _ dictionary object reference where to look up
    #param3 _ mode (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #              (                  3-unsupervised :: 4-unkown words on training)    

 
    my $rwin = shift;
    my $rdict = shift;
    my $mode = shift;

    if ($mode == $COMMON::mode3) { return $rdict->unknown_word($rwin->get_core_actual_word()); }
    else { return $rdict->unknown_word($rwin->get_core_word()); }

    #return $rdict->unknown_word($rwin->get_core_word());
}

sub sfp_window{
    #description _ returns true only if the window core word POS is a 
    #              sentence-final punctuation one.
    #param1 _ window reference

    my $rwin = shift;

    return (COMMON::end_of_sentence($rwin->get_core_word()));
}

sub get_sentence_info
{
   #description _ some sentence general information is retrieved.
   #param1 _ CORPUS filename
   #param2 _ text direction "LR" for left-to-right ; "RL" for right-to-left
   #param3 _ current token being read
   #@return - punctuation ([.?!]) 

   my $CORPUS = shift;
   my $direction = shift;
   my $token = shift;

   my @info;

   my $current = $CORPUS->getpos;

   my $lastw = $token;

   if ($direction eq $COMMON::lrmode) { # left-to-right (as usual)
      # last word ------------------------------------
      my $stop = 0;
      my @entry;
      while ((my $line = $CORPUS->getline()) and (!$stop)) {
         chomp($line);      
         @entry = split($COMMON::in_valseparator, $line);
         if (COMMON::end_of_sentence($entry[0])) { $stop = 1; }
      }
      # ----------------------------------------------
      if ($stop) { $lastw = $entry[0]; }
   }

   $CORPUS->setpos($current);

   push(@info, $lastw);
  
   return \@info;
}

sub get_sentence_info_list
{
   #description _ some sentence general information is retrieved.
   #param1 _ CORPUS list reference
   #param2 _ text direction "LR" for left-to-right ; "RL" for right-to-left
   #param3 _ current position being read
   #@return - punctuation ([.?!]) 

   my $LIST = shift;
   my $direction = shift;
   my $i = shift;

   my @entry = split(/$COMMON::in_valseparator/, $LIST->[$i]);
   my $lastw = $entry[0];

   if ($direction eq $COMMON::lrmode) { # left-to-right (as usual)
      # last word ------------------------------------
      my $stop = 0;
      my $SIZE = scalar(@{$LIST});
      while (($i < $SIZE) and (!$stop)) {
         @entry = split($COMMON::in_valseparator, $LIST->[$i]);
         if (COMMON::end_of_sentence($entry[0])) { $stop = 1; }
         $i++;
      }
      if ($stop) { $lastw = $entry[0]; }
   }

   my @info;
   push(@info, $lastw);
  
   return \@info;
}


sub get_sentence_info_list_entry
{
   #description _ some sentence general information is retrieved.
   #param1 _ CORPUS entry list reference
   #param2 _ text direction "LR" for left-to-right ; "RL" for right-to-left
   #param3 _ current position being read
   #@return - punctuation ([.?!]) 

   my $LIST = shift;
   my $direction = shift;
   my $i = shift;

   my $lastw = $LIST->[$i]->get_word;

   if ($direction eq $COMMON::lrmode) { # left-to-right (as usual)
      # last word ------------------------------------
      my $stop = 0;
      my $SIZE = scalar(@{$LIST});
      my $w;
      while (($i < $SIZE) and (!$stop)) {
         $w = $LIST->[$i]->get_word;
         if (COMMON::end_of_sentence($w)) { $stop = 1; }
         $i++;
      }
      if ($stop) { $lastw = $w; }
   }

   my @info;
   push(@info, $lastw);
  
   return \@info;
}

# ============================== SAMPLE EXTRACTION ====================================

sub do_attribs_kn
{
   #description _ creates a convenient set of samples (attrib:value list) out from a given corpus
   #param1 _ CORPUS filename
   #param2 _ DICTIONARY object reference
   #param3 _ target SAMPLE SET filename
   #param4 _ text direction "LR" for left-to-right ; "RL" for right-to-left
   #param5 _ mode (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
   #              (                  3-unsupervised :: 4-unkown words on training)    
   #param6 _ configuration hash reference
   #param7 _ report file
   #param8 _ verbosity [0..3]
   #@return _ number of positive and negative examples

   my $corpus = shift;
   my $rdict = shift;
   my $smplset = shift;
   my $direction = shift;
   my $mode = shift;
   my $config = shift;
   my $report = shift;
   my $verbose = shift;
   my $Unihan = shift;
   my $BS = shift;

   my $CORPUS = new IO::File("< $corpus") or die "Couldn't open input file: $corpus\n";
   my $SMPLSET = new IO::File("> $smplset") or die "Couldn't open output file: $smplset\n";
   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "FEATURE EXTRACTION from < $corpus > onto < $smplset > ..."); }
   if ($verbose > $COMMON::verbose2) { COMMON::report($report, "\n"); }

   my $fs;
   if ($mode == $COMMON::mode0) {
      $fs = SVMTAGGER::read_fs_list($config->{A0k}, $config->{wlen}, $config->{wcore});
   }
   elsif ($mode == $COMMON::mode1) {
      $fs = SVMTAGGER::read_fs_list($config->{A1k}, $config->{wlen}, $config->{wcore});
   }
   elsif ($mode == $COMMON::mode2) {
      $fs = SVMTAGGER::read_fs_list($config->{A2k}, $config->{wlen}, $config->{wcore});
   }
   elsif ($mode == $COMMON::mode3) {
      $fs = SVMTAGGER::read_fs_list($config->{A3k}, $config->{wlen}, $config->{wcore});
   }
   else { die "BAD MODE [$mode]\n"; }   

   #sliding window generation
   my $rwindow = new SWINDOW($config->{wlen}, $config->{wcore});

   my $sinfo;
   if ($direction eq $COMMON::lrmode) { $sinfo = get_sentence_info($CORPUS, $direction, ""); }
   my $iter = 0;

   while (defined(my $line = $CORPUS->getline())) {
       chomp($line);

       my @entry = split($COMMON::in_valseparator, $line);
       if (COMMON::end_of_sentence($entry[0])) {
          $sinfo = get_sentence_info($CORPUS, $direction, $entry[0]);
       }

       $iter++;
       if ($verbose > $COMMON::verbose1) { COMMON::show_progress_stdout($iter, $COMMON::progress1, $COMMON::progress2); }

       #shift sliding window one position left
       $rwindow->lshift(1);
       #push current entry onto the sliding window
       my @columns = @entry; shift(@columns); shift(@columns);
       $rwindow->push($entry[0], (scalar(@entry) > 1)? $entry[1] : $rdict->tag($entry[0]), $config->{wlen} - 1, \@columns);

       #print STDERR Dumper($rwindow);

       #is the core item active?
       if ($rwindow->active()) {
          #is the core word ambiguous?
          #$rwindow->print;
          if ((($mode == $COMMON::mode3) and (!(ambiguous_word_window($rwindow, $rdict, $mode)))) or (($mode != $COMMON::mode3) and (ambiguous_word_window($rwindow, $rdict, $mode)))) {
             #window preparation
             my $rattribw = $rwindow->prepare($direction); 
             #window attribute generation
             my $rattribs = ATTGEN::generate_features($rattribw, $rdict, $sinfo, $mode, $fs,$Unihan,$BS);# have modified  add $Unihan
             #write sample onto file
             SVM::write_sample($rwindow->get_core_word().$COMMON::attvalseparator.$rwindow->get_core_pos(), $rattribs, $SMPLSET);
	  }
       }
   }

   my $i = $config->{wlen} - $config->{wcore} - 1;
   while ($i > 0) { #process last words     
      $rwindow->lshift(1);
      if ($rwindow->active()) {
         if ((($mode == $COMMON::mode3) and (!(ambiguous_word_window($rwindow, $rdict, $mode)))) or (($mode != $COMMON::mode3) and (ambiguous_word_window($rwindow, $rdict, $mode)))) {
            my $rattribw = $rwindow->prepare($direction); 
            my $rattribs = ATTGEN::generate_features($rattribw, $rdict, $sinfo, $mode, $fs,$Unihan,$BS);# have modified  add $Unihan
            SVM::write_sample($rwindow->get_core_word().$COMMON::attvalseparator.$rwindow->get_core_pos(), $rattribs, $SMPLSET);
         }
      }
      $i--;
   }

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "...$iter WORDS [DONE]\n"); }

   $CORPUS->close();
   $SMPLSET->close();
}

sub do_attribs_kn_unk
{
   #description _ creates a convenient set of samples (attrib:value list) out from a given corpus
   #              for unknown words
   #param1 _ MODEL NAME (WSJ / WSJTP)
   #param2 _ TRAINING CORPUS filename
   #param3 _ target SAMPLE SET filename
   #param4 _ text direction "LR" for left-to-right ; "RL" for right-to-left
   #param5 _ mode (generate_features 4-ambiguous-right ++ unknown words on training)    
   #param6 _ configuration hash reference
   #param7 _ report file
   #param8 _ verbosity [0..3]

   my $model = shift;
   my $corpus = shift;
   my $smplset = shift;
   my $direction = shift;
   my $mode = shift;
   my $config = shift;
   my $report = shift;
   my $verbose = shift;
   my $Unihan = shift;
   my $BS = shift;

   my $fambp = $model.".".$COMMON::AMBPEXT;
   my $funkp = $model.".".$COMMON::UNKPEXT;

   my $CORPUS = new IO::File("< $corpus") or die "Couldn't open input file: $corpus\n";
   my $SMPLSET = new IO::File("> $smplset") or die "Couldn't open output file: $smplset\n";
   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "FEATURE EXTRACTION from < $corpus > onto < $smplset > ..."); }
   if ($verbose > $COMMON::verbose2) { COMMON::report($report, "\n"); }

   srand();
   my $fs;
   if ($mode == $COMMON::mode4) { $fs = SVMTAGGER::read_fs_list($config->{A4k}, $config->{wlen}, $config->{wcore}); }
   else { die "BAD MODE [$mode]\n"; }   

   my $dict = $model.".".$COMMON::DICTEXT.".".rand(100000);
   my $DICT = $dict.".R";
   my $rdict;
   #sliding window generation
   my $rwindow = new SWINDOW($config->{wlen}, $config->{wcore});

   my ($trainsize, $nchunks) = SVMTAGGER::get_train_size($corpus, $config->{X});
   if ($verbose > $COMMON::verbose2) { print "CHUNKSIZE = $trainsize\nnCHUNKS = $nchunks\n"; }
   my $sinfo;
   if ($direction eq $COMMON::lrmode) { $sinfo = SVMTAGGER::get_sentence_info($CORPUS, $direction, ""); }
   my $iter = 0;
   my $chunk = 0;
   while (defined(my $line = $CORPUS->getline())) {
      chomp($line);

      my @entry = split($COMMON::in_valseparator, $line);
      if (COMMON::end_of_sentence($entry[0])) {
         $sinfo = get_sentence_info($CORPUS, $direction, $entry[0]);
      }

      if (($iter % $trainsize) == 0) {
	 if ($verbose > $COMMON::verbose2) { print "STARTING CHUNK $chunk : reading word -> $iter\n"; }
         #dictionary generation
         SVMTAGGER::generate_dictionary($corpus, $dict, $chunk * $trainsize, ($chunk + 1) * $trainsize - 1);
         my $N;
         if ($mode != $COMMON::mode3) {
            #dictionary repairing
            if ($config->{R} ne "") {
               $N = SVMTAGGER::repair_dictionary($dict, $config->{R}, $DICT);
	    }
            else {
               $N = SVMTAGGER::repair_dictionary_H($dict, $DICT, $config->{Dratio});
	    }
            system "mv $DICT $dict";
            #dictionary enriching
            if ($config->{BLEX} ne "") { SVMTAGGER::enrich_dictionary($dict, $config->{BLEX}); }
	 }
         $rdict = new DICTIONARY($dict, $fambp, $funkp);
         system "rm -f $dict";
         if ($verbose > $COMMON::verbose2) { print " -> number of words $rdict->{nwords} [#REPAIRINGS = $N]\n"; }
	 $chunk++;   
      }

      #shift sliding window one position left
      $rwindow->lshift(1);
      #push current entry onto the sliding window
      my @columns = @entry; shift(@columns); shift(@columns);
      $rwindow->push($entry[0], (scalar(@entry) > 1)? $entry[1] : $rdict->tag($entry[0]), $config->{wlen} - 1, \@columns);

      #is the core item active?
      if ($rwindow->active()) {
         #is the core word ambiguous?
         if (ambiguous_word_window($rwindow, $rdict, $mode)) {
            #window preparation
            my $rattribw = $rwindow->prepare($direction); 
            #window attribute generation
            my $rattribs = ATTGEN::generate_features($rattribw, $rdict, $sinfo, $mode, $fs,$Unihan,$BS);# have modified  add $Unihan
            #write sample onto file
            SVM::write_sample($rwindow->get_core_word().$COMMON::attvalseparator.$rwindow->get_core_pos(), $rattribs, $SMPLSET);
         }
      }

      $iter++;
   }

   my $i = $config->{wlen} - $config->{wcore} - 1;
   while ($i > 0) { #process last words     
      $rwindow->lshift(1);
      if ($rwindow->active()) {
         if (ambiguous_word_window($rwindow, $rdict, $mode)) {
            my $rattribw = $rwindow->prepare($direction); 
            my $rattribs = ATTGEN::generate_features($rattribw, $rdict, $sinfo, $mode, $fs,$Unihan,$BS);# have modified  add $Unihan
            SVM::write_sample($rwindow->get_core_word().$COMMON::attvalseparator.$rwindow->get_core_pos(), $rattribs, $SMPLSET);
         }
      }
      $i--;
   }

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "...$iter WORDS [DONE]\n"); }

   $CORPUS->close();
   $SMPLSET->close();
}

sub do_attribs_unk
{
   #description _ creates a convenient set of samples (attrib:value list) out from a given corpus
   #              for unknown words
   #param1 _ MODEL NAME (WSJ / WSJTP)
   #param2 _ TRAINING CORPUS filename
   #param3 _ DICTIONARY object reference
   #param4 _ target SAMPLE SET filename
   #param5 _ text direction "LR" for left-to-right ; "RL" for right-to-left
   #param6 _ mode (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
   #              (                  3-unsupervised :: 4-unkown words on training)    
   #param7 _ configuration hash reference
   #param8 _ report file
   #param9 _ verbosity [0..3]

   my $model = shift;
   my $corpus = shift;
   my $rlex = shift;
   my $smplset = shift;
   my $direction = shift;
   my $mode = shift;
   my $config = shift;
   my $report = shift;
   my $verbose = shift;
   my $Unihan = shift;
   my $BS = shift;

   my $fambp = $model.".".$COMMON::AMBPEXT;
   my $funkp = $model.".".$COMMON::UNKPEXT;

   my $CORPUS = new IO::File("< $corpus") or die "Couldn't open input file: $corpus\n";
   my $SMPLSET = new IO::File("> $smplset") or die "Couldn't open output file: $smplset\n";
   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "FEATURE EXTRACTION from < $corpus > onto < $smplset > ..."); }
   if ($verbose > $COMMON::verbose2) { COMMON::report($report, "\n"); }

   srand();
   my $fs;
   if ($mode == $COMMON::mode0) {
      $fs = SVMTAGGER::read_fs_list($config->{A0u}, $config->{wlen}, $config->{wcore});
   }
   elsif ($mode == $COMMON::mode1) {
      $fs = SVMTAGGER::read_fs_list($config->{A1u}, $config->{wlen}, $config->{wcore});
   }
   elsif ($mode == $COMMON::mode2) {
      $fs = SVMTAGGER::read_fs_list($config->{A2u}, $config->{wlen}, $config->{wcore});
   }
   elsif ($mode == $COMMON::mode3) {
      $fs = SVMTAGGER::read_fs_list($config->{A3u}, $config->{wlen}, $config->{wcore});
   }
   elsif ($mode == $COMMON::mode4) {
      $fs = SVMTAGGER::read_fs_list($config->{A4u}, $config->{wlen}, $config->{wcore});
   }
   else { die "BAD MODE [$mode]\n"; }   
 
   my $dict = $model.".".$COMMON::DICTEXT.".".rand(100000);
   my $DICT = $dict.".R";
   my $rdict;
   #sliding window generation
   my $rwindow = new SWINDOW($config->{wlen}, $config->{wcore});
   my ($trainsize, $nchunks) = SVMTAGGER::get_train_size($corpus, $config->{X});
   if ($verbose > $COMMON::verbose2) { print "X = ", $config->{X}, " :: CHUNKSIZE = $trainsize :: CHUNKS = $nchunks\n"; }
   my $sinfo;
   if ($direction eq $COMMON::lrmode) { $sinfo = SVMTAGGER::get_sentence_info($CORPUS, $direction, ""); }
   my $iter = 0;
   my $chunk = 0;
   while (defined(my $line = $CORPUS->getline())) {
      chomp($line);

      my @entry = split($COMMON::in_valseparator, $line);
      if (COMMON::end_of_sentence($entry[0])) {
         $sinfo = get_sentence_info($CORPUS, $direction, $entry[0]);
      }

      if (($iter % $trainsize) == 0) {
	 if ($verbose > $COMMON::verbose2) { print "STARTING CHUNK $chunk : reading word -> $iter\n"; }
	 elsif ($verbose > $COMMON::verbose1) { print "."; }
         #dictionary generation
         SVMTAGGER::generate_dictionary($corpus, $dict, $chunk * $trainsize, ($chunk + 1) * $trainsize - 1);
         my $N = 0;
         if ($mode != $COMMON::mode3) {
            #dictionary repairing
            if ($config->{R} ne "") {
               $N = SVMTAGGER::repair_dictionary($dict, $config->{R}, $DICT);
	    }
            else {
               $N = SVMTAGGER::repair_dictionary_H($dict, $DICT, $config->{Dratio});
	    }
            system "mv $DICT $dict";
	 }
         else {
            #dictionary enriching [unsupervised]
            if ($config->{LEX} ne "") { SVMTAGGER::enrich_dictionary($dict, $config->{LEX}); }
	 }
         #dictionary enriching
         if ($config->{BLEX} ne "") { SVMTAGGER::enrich_dictionary($dict, $config->{BLEX}); }
         $rdict = new DICTIONARY($dict, $fambp, $funkp);
         system "rm -f $dict";
         if ($verbose > $COMMON::verbose2) { print " -> number of words $rdict->{nwords} [#REPAIRINGS = $N]\n"; }
	 $chunk++;
      }


      #shift sliding window one position left
      $rwindow->lshift(1);
      #push current entry onto the sliding window
      my @columns = @entry; shift(@columns); shift(@columns);
      $rwindow->push($entry[0], (scalar(@entry) > 1)? $entry[1] : $rlex->tag($entry[0]), $config->{wlen} - 1, \@columns);

      #print STDERR Dumper($rwindow);

      #is the core item active?
      if ($rwindow->active()) {
         #is the core word unknown?
	 if (unknown_word_window($rwindow, $rdict, $mode)) {
            if (($mode == $COMMON::mode3)? !(ambiguous_word_window($rwindow, $rlex, $mode)) : 1) {
               #window preparation
               my $rattribw = $rwindow->prepare($direction); 
               #window attribute generation
               my $rattribs = ATTGEN::generate_features($rattribw, $rdict, $sinfo, $mode, $fs,$Unihan,$BS);# have modified  add $Unihan
               #write sample onto file
               SVM::write_sample($rwindow->get_core_word().$COMMON::attvalseparator.$rwindow->get_core_pos(), $rattribs, $SMPLSET);
            }
         }
      }

      $iter++;
   }

   my $i = $config->{wlen} - $config->{wcore} - 1;
   while ($i > 0) { #process last words     
      $rwindow->lshift(1);
      if ($rwindow->active()) {
	 if (unknown_word_window($rwindow, $rdict, $mode)) {
            if (($mode == $COMMON::mode3)? !(ambiguous_word_window($rwindow, $rlex, $mode)) : 1) {
               my $rattribw = $rwindow->prepare($direction);
               my $rattribs = ATTGEN::generate_features($rattribw, $rdict, $sinfo, $mode, $fs,$Unihan,$BS);# have modified  add $Unihan
               SVM::write_sample($rwindow->get_core_word().$COMMON::attvalseparator.$rwindow->get_core_pos(), $rattribs, $SMPLSET);
            }
         }
      }
      $i--;
   }

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "...$iter WORDS [DONE]\n"); }

   $CORPUS->close();
   $SMPLSET->close();
}

# ======================================================================================
# ========================== LEARNING/CLASSIFYING/TAGGING ==============================
# ======================================================================================

# ===================================== LEARN =========================================

sub dress_naked_set_train{
    #description _ performs example selection of a DSF file
    #              given a         <+1|-1> {attrib:val}*        file
    #param1 _ dictionary object reference
    #param2 _ naked set filename
    #param3 _ current POS being processed
    #param4 _ dressed dataset filename (onto store the dressed samples for POS SVM)

    my $rdict = shift;
    my $nakedsmpl = shift;
    my $pos = shift;
    my $dataset = shift;

    my $Npositive = 0;
    my $Nnegative = 0;

    my $NAKEDSMPL = new IO::File("< $nakedsmpl") or die "Couldn't open input file: $nakedsmpl\n";
    my $DATASET = new IO::File("> $dataset") or die "Couldn't open output file: $dataset\n";

    while (defined(my $entry = $NAKEDSMPL->getline())) {

	my @line = split(/$COMMON::pairseparator/, $entry);
        my $len = @line;
        my @WP = split(/$COMMON::attvalseparator/, $line[0]);
        my $sampleword = $WP[0];
        my $samplepos = $WP[1];

        my $new_entry;
        if ($samplepos eq $pos) { $new_entry = "+1"; }
        else { $new_entry = "-1"; }
        my $i = 1;
        while ($i < $len) {
	   $new_entry = $new_entry.$COMMON::pairseparator.$line[$i];
	   $i++;
	}

        my $amb = $rdict->get_potser($sampleword);

        foreach my $Ci (@{$amb}) {
            #write only possible POSs
	    if ($Ci eq $pos) {
              print $DATASET $new_entry;
              if ($samplepos eq $pos) { $Npositive++; }
              else { $Nnegative++; }
            }
        }
    }

    $NAKEDSMPL->close();
    $DATASET->close();

    return($Npositive, $Nnegative);
}

sub dress_naked_set_train_unsupervised{
    #description _ given a POS, transforms a naked set
    #                       <POS> {attrib:val}*
    #              to a smv_classify format for training
    #                       <+1|-1> {attrib:val}*
    #param1 _ dictionary object reference
    #param2 _ equivalence classes hash reference
    #param3 _ naked set filename
    #param4 _ current POS being processed
    #param5 _ dressed dataset filename (onto store the dressed samples for POS SVM)
    #@return _ number of positive and negative examples

    my $rdict = shift;
    my $eqC = shift;
    my $nakedsmpl = shift;
    my $pos = shift;
    my $dataset = shift;

    my $Npositive = 0;
    my $Nnegative = 0;

    my $NAKEDSMPL = new IO::File("< $nakedsmpl") or die "Couldn't open input file: $nakedsmpl\n";
    my $DATASET = new IO::File("> $dataset") or die "Couldn't open output file: $dataset\n";

    while (defined(my $entry = $NAKEDSMPL->getline())) {
	my @line = split(/$COMMON::pairseparator/, $entry);
        my $len = @line;
        my @WP = split(/$COMMON::attvalseparator/, $line[0]);
        my $sampleword = $WP[0];
        my $samplepos = $WP[1];

        my $new_entry;
        my $i = 1;
        while ($i < $len) {
	   $new_entry = $new_entry.$COMMON::pairseparator.$line[$i];
	   $i++;
	}

        if ($samplepos eq $pos) {
           $new_entry = "+1".$new_entry;
           print $DATASET $new_entry;
           $Npositive++;
        }
        else {
          #if (1) {
          if ($rdict->frequent_word($sampleword)) {
           #print STDERR Dumper($eqC->{$samplepos});
           foreach my $Ci (@{$eqC->{$samplepos}}) {
	      #print STDERR "$Ci vs. $pos\n";
 	      if ($Ci eq $pos) {
                 $new_entry = "-1".$new_entry;
                 print $DATASET $new_entry;
                 $Nnegative++;
	      }
	   }
          }
        }        
    }

    $NAKEDSMPL->close();
    $DATASET->close();

    return($Npositive, $Nnegative);
}

sub dress_naked_set_train_unk{
    #description _ given a POS, transforms a naked set
    #                       <POS> {attrib:val}*
    #              to a smv_classify format for training
    #                       <+1|-1> {attrib:val}*
    #param1 _ naked set filename
    #param2 _ current POS being processed
    #param3 _ dressed dataset filename (onto store the dressed samples for POS SVM)
    #@return _ number of positive and negative examples

    my $nakedsmpl = shift;
    my $pos = shift;
    my $dataset = shift;

    my $Npositive = 0;
    my $Nnegative = 0;

    my $NAKEDSMPL = new IO::File("< $nakedsmpl") or die "Couldn't open input file: $nakedsmpl\n";
    my $DATASET = new IO::File("> $dataset") or die "Couldn't open output file: $dataset\n";

    while (defined(my $entry = $NAKEDSMPL->getline())) {

	my @line = split(/$COMMON::pairseparator/, $entry);
        my $len = @line;
        my @WP = split(/$COMMON::attvalseparator/, $line[0]);
        my $samplepos = $WP[1];

        my $new_entry;
        if ($samplepos eq $pos) { $new_entry = "+1"; $Npositive++; }
        else { $new_entry = "-1"; $Nnegative++; }
        my $i = 1;
        while ($i < $len) {
	   $new_entry = $new_entry.$COMMON::pairseparator.$line[$i];
	   $i++;
	}

        print $DATASET $new_entry;
    }

    $NAKEDSMPL->close();
    $DATASET->close();

    return($Npositive, $Nnegative);
}

sub do_learn
{
   #description _ svm learning of POS-tagging for known words.
   #param1  _ dictionary object reference   
   #param2  _ SAMPLE MAPPING filename       
   #param3  _ AMBIGUOUS POS list reference  
   #param4  _ MODEL NAME                    
   #param5  _ model type file EXTENSION     
   #param6  _ model direction file EXTENSION
   #param7  _ TRAINING SET filename         
   #param8  _ C parameter                   
   #param9  _ kernel type                   
   #param10 _ kernel degree                 
   #param11 _ mode                          
   #param12 _ SVM-light directory (Joachims software) 
   #param13 _ Eratio (to compute classes of equivalence in unsupervised mode)  
   #param14 _ REPORT filename               
   #param15 _ verbosity

   my $rdict = shift;
   my $smplmap = shift;
   my $rambpos = shift;
   my $model = shift;
   my $modext = shift;
   my $dirext = shift;
   my $traindsf = shift;
   my $C = shift;
   my $kernel = shift;
   my $degree = shift;
   my $mode = shift;
   my $svmdir = shift;
   my $Eratio = shift;
   my $report = shift;
   my $verbose = shift;

   my $options;
   if ($C != 0) { $options = " -c $C"; }
   if ($kernel != 0) { $options .= " -t $kernel"; }
   if ($degree != 0) { $options .= " -d $degree"; }

   # -- intermediate common files
   srand();

   my $Npositive;
   my $Nnegative;

   my $eqCs;

   if ($mode == $COMMON::mode3) {
      if ($verbose > $COMMON::verbose1) { COMMON::report($report, "EXTRACTING EQUIVALENCE CLASSES [ratio = $Eratio]...\n"); }
      $eqCs = $rdict->determine_eq_classes($rambpos, $Eratio);
      if ($verbose > $COMMON::verbose2) {
         foreach my $e (sort keys %{$eqCs}) {
            COMMON::report($report, $e." \t: \t".join(" ", @{$eqCs->{$e}})."\n");
         }
      }
   }

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "LEARNING SVM MODELS... [C = ".($C + 0)."] "); }
   if ($verbose > $COMMON::verbose2) { COMMON::report($report, "\n"); }

   my $timeA = new Benchmark;

   my $ID = 0;
   while ($ID < scalar(@{$rambpos})) {
      my $tag = $rambpos->[$ID];

      my $time1 = new Benchmark;

      my $currenttraindsf = "$model.$modext.$dirext.$COMMON::smplext.$ID.$COMMON::DSFEXT.".rand(100000);

      #PREPARING 4 TRAINING
      if ($verbose == $COMMON::verbose2) {
         COMMON::report($report, "..$tag");
      }
      elsif ($verbose > $COMMON::verbose2) {
         print "********************************************************************************************\nPREPARING TRAINING SET for < $tag >...\n";
      }
      if ($mode == $COMMON::mode3) {      
         ($Npositive, $Nnegative) = dress_naked_set_train_unsupervised($rdict, $eqCs, $traindsf, $tag, $currenttraindsf);
      }
      else {
         ($Npositive, $Nnegative) = dress_naked_set_train($rdict, $traindsf, $tag, $currenttraindsf); 
      }

      #TRAINING
      if ($verbose > $COMMON::verbose2) { COMMON::report($report, "LEARNING < $tag > (OPTIONS: $options) [#EXAMPLES = ".($Npositive + $Nnegative)." :: (+) ".$Npositive." :: (-) ".$Nnegative."]\n"); }
      SVM::svm_learn($model, $currenttraindsf, $ID, $options, "$modext.$dirext.$COMMON::SVMEXT", $svmdir, $report, $verbose);
      my $time2 = new Benchmark;
      if ($verbose > $COMMON::verbose2) { COMMON::print_benchmark_file($time1, $time2, $report); }
      system "rm -f $currenttraindsf";
      $ID++;
   }

   my $timeB = new Benchmark;

   if ($verbose == $COMMON::verbose2) { COMMON::report($report, " [DONE]\n"); }

   if ($verbose > $COMMON::verbose1) { COMMON::print_benchmark_file($timeA, $timeB, $report); }
}

sub do_learn_unk
{
   #description _ svm learning of POS-tagging for unknown words.
   #param1  _ SAMPLE MAPPING filename       
   #param2  _ UNKNOWN POS list reference    
   #param3  _ MODEL NAME
   #param4  _ model type file EXTENSION     
   #param5  _ model direction file EXTENSION
   #param6  _ TRAINING SET filename         
   #param7  _ C parameter                   
   #param8  _ kernel type                   
   #param9  _ kernel degree                 
   #param10 _ mode                          
   #param11 _ SVM-light directory (Joachims software) 
   #param12 _ REPORT filename
   #param13 _ verbosity

   my $smplmap = shift;
   my $runkpos = shift;
   my $model = shift;
   my $modext = shift;
   my $dirext = shift;
   my $traindsf = shift;
   my $C = shift;
   my $kernel = shift;
   my $degree = shift;
   my $mode = shift;
   my $svmdir = shift;
   my $report = shift;
   my $verbose = shift;

   my $options = "";
   if ($C != 0) { $options = "-c $C"; }
   if ($kernel != 0) { $options .= " -t $kernel"; }
   if ($degree != 0) { $options .= " -d $degree"; }

   # -- intermediate common files
   srand();

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "LEARNING SVM MODELS... [C = ".($C + 0)."] "); }
   if ($verbose > $COMMON::verbose2) { COMMON::report($report, "\n"); }

   my $timeA = new Benchmark;

   my $ID = 0;
   while ($ID < scalar(@{$runkpos})) {
      my $tag = $runkpos->[$ID];

      my $time1 = new Benchmark;
      my $currenttraindsf = "$model.$modext.$dirext.$COMMON::smplext.$ID.$COMMON::DSFEXT.".rand(100000);

      if ($verbose == $COMMON::verbose2) {
         COMMON::report($report, "..$tag");
      }
      elsif ($verbose > $COMMON::verbose2) {
         print "********************************************************************************************\nPREPARING TRAINING SET for < $tag >...\n";
      }
      my ($Npositive, $Nnegative) = dress_naked_set_train_unk($traindsf, $tag, $currenttraindsf);

      #TRAINING
      if ($verbose > $COMMON::verbose2) { COMMON::report($report, "LEARNING < $tag > (OPTIONS: $options) [#EXAMPLES = ".($Npositive + $Nnegative)." :: (+) ".$Npositive." :: (-) ".$Nnegative."]\n"); }
      SVM::svm_learn($model, $currenttraindsf, $ID, $options, "$COMMON::unkext.$modext.$dirext.$COMMON::SVMEXT", $svmdir, $report, $verbose);
      my $time2 = new Benchmark;
      if ($verbose > $COMMON::verbose2) { COMMON::print_benchmark_file($time1, $time2, $report); }
      system "rm -f $currenttraindsf";
      $ID++;
   }

   my $timeB = new Benchmark;

   if ($verbose == $COMMON::verbose2) { COMMON::report($report, " [DONE]\n"); }

   if ($verbose > $COMMON::verbose1) { COMMON::print_benchmark_file($timeA, $timeB, $report); }
}

# =================================== CLASSIFY ========================================

sub dress_naked_set_classify{
    #description _ given a POS, transforms a naked set
    #                       <POS> {attrib:val}*
    #              to a smv_classify format for classifying
    #                       <+1|-1> {attrib:val}*
    #param1 _ naked set filename
    #param2 _ current POS being processed
    #param3 _ dressed dataset filename (onto store the dressed samples for POS SVM)
    #@return _ number of samples in the set

    my $nakedsmpl = shift;
    my $pos = shift;
    my $dataset = shift;

    my $N = 0;

    my $NAKEDSMPL = new IO::File("< $nakedsmpl") or die "Couldn't open input file: $nakedsmpl\n";
    my $DATASET = new IO::File("> $dataset") or die "Couldn't open output file: $dataset\n";

    my $samplepos;

    while (defined(my $entry = $NAKEDSMPL->getline())) {
	my @line = split(/$COMMON::pairseparator/, $entry);
        my $len = @line;
        $samplepos = $line[0];

        my $new_entry;
        if ($samplepos eq $pos) { $new_entry = "+1"; }
        else { $new_entry = "-1"; }
        my $i = 1;
        while ($i < $len) {
           $new_entry = $new_entry.$COMMON::pairseparator.$line[$i];
	   $i++;
	}

        print $DATASET $new_entry;
        $N++;
    }

    $NAKEDSMPL->close();
    $DATASET->close();

    return($N);
}

# ======================================================================================
# ================================= SVM TAGGER =========================================
# ======================================================================================


sub get_train_size
{
   #description _ figures out what the size of the training chunk must be so as to
   #              comprise a given percentage of unknown words
   #param1 _ TRAINING SET filename                       
   #param2 _ unknown word percentage                     
   #@return _ (chunk size, nchunks)                      

   my $corpus = shift;
   my $X = shift;

   # read the corpus
   my $nwords;
   my $ndwords;
   my %WORDS;
   my $CORPUS = new IO::File("< $corpus") or die "Couldn't open input file: $corpus\n";
   while (defined(my $line = $CORPUS->getline())) {
       $nwords++;
       my @entry = split($COMMON::in_valseparator, $line);
       if (exists($WORDS{$entry[0]})) { $WORDS{$entry[0]} += 1; }
       else { $WORDS{$entry[0]} = 1; $ndwords++; }
   }
   $CORPUS->close();

   # read again until a certain point where X is met   --> $ndwords * (100 - $X) / 100;  
   my $meeting = $ndwords * (100 - $X) / 100;
   my $nwords2;
   my $ndwords2;
   my %WORDS2;
   $CORPUS = new IO::File("< $corpus") or die "Couldn't open input file: $corpus\n";
   while (defined(my $line = $CORPUS->getline()) and ($ndwords2 < $meeting)) {
       $nwords2++;
       my @entry = split($COMMON::in_valseparator, $line);
       if (exists($WORDS2{$entry[0]})) { $WORDS2{$entry[0]} += 1; }
       else { $WORDS2{$entry[0]} = 1; $ndwords2++; }
   }
   $CORPUS->close();

   my $chunks = 1; my $size = $nwords;
   if ($nwords - $nwords2 > 0) {
      $chunks = sprintf("%d", $nwords / ($nwords - $nwords2));
      $size = sprintf("%d", $nwords / $chunks) + 1;
   }
   return ($size, $chunks);
}


# ================================= TAGGING KN + UNK ====================================
# ---------------------------------------------------------------------------------------
# =================================== merged models =====================================
# =======================================================================================

sub load_models
{
    #description _ responsible for loading svm models in a certain manner.
    #param1 _ MODEL NAME                 
    #param2 _ EPSILON threshold for W features relevance  [KNOWN WORDS] 
    #param3 _ OMEGA threshold for W features relevance  [UNKNOWN WORDS] 
    #param4 _ VERBOSE 
    #param5 _ text direction "LR" for left-to-right ; "RL" for right-to-left; "LRL" for both
    #param6 _ mode (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #              (                  3-unsupervised :: 4-unkown words on training)    
    #param7  _ ambiguous pos list reference
    #param8  _ unknown pos list reference
    #param9  _ feature set list reference for KNOWN WORDS
    #param10 _ feature set list reference for UNKNOWN WORDS
    #@return _ SVMT models

    my $corpus = shift;
    my $epsilon = shift;
    my $omega = shift;
    my $verbose = shift;
    my $direction = shift;
    my $mode = shift;
    my $rambp = shift;
    my $runkp = shift;
    my $fsk = shift;
    my $fsu = shift;

    if ($verbose > $COMMON::verbose1) {
       print STDERR "READING MODELS < DIRECTION = ";
       if ($direction eq $COMMON::lrmode) { print STDERR "left-to-right"; }
       elsif ($direction eq $COMMON::rlmode) { print STDERR "right-to-left"; }
       elsif ($direction eq $COMMON::lrlmode) { print STDERR "left-to-right then right-to-left"; }
       elsif ($direction eq $COMMON::glrlmode) { print STDERR "left-to-right then right-to-left (global)"; }
       else { print STDERR "$direction"; }
       print STDERR " :: MODEL = ";
       if ($mode == $COMMON::mode0) { print STDERR "ambiguous context"; }
       elsif ($mode == $COMMON::mode1) { print STDERR "disambiguated context"; }
       elsif ($mode == $COMMON::mode2) { print STDERR "no context"; }
       elsif ($mode == $COMMON::mode3) { print STDERR "unsupervised"; }
       elsif ($mode == $COMMON::mode4) { print STDERR "ambiguous context [unknown words on training]"; }
       else { print STDERR $mode; }
       print STDERR " >\n";
    }

    my $dirext;
    if ($direction eq $COMMON::lrmode) {
       $dirext .= "$COMMON::lrext";
    }
    elsif ($direction eq $COMMON::rlmode) {
       $dirext .= "$COMMON::rlext";
    }
    my $modext = SVMTAGGER::find_mext($mode);

    #SAMPLE MAPPING  (in)
    my $smplmap = $corpus.".".$modext.".".$dirext.".".$COMMON::smplext.".".$COMMON::mapext;
    #UNKNOWN WORDS MAPPING (in)
    my $unksmplmap = $corpus.".".$COMMON::unkext.".".$modext.".".$dirext.".".$COMMON::smplext.".".$COMMON::mapext;

    if ($verbose > $COMMON::verbose1) { print STDERR "(1) READING MODELS (weights and biases) FOR KNOWN WORDS <$corpus.$modext.$dirext.$COMMON::MRGEXT>...\n"; }
    my ($KNMODEL, $rB, $Nk, $Nkout) = SVM::read_merged_models($corpus.".".$modext.".".$dirext.".".$COMMON::MRGEXT, $epsilon, $verbose);
    if ($verbose > $COMMON::verbose1) { print STDERR "(2) READING MODELS (weights and biases) FOR UNKNOWN WORDS <$corpus.$COMMON::unkext.$modext.$dirext.$COMMON::MRGEXT>...\n"; }
    my ($UNKMODEL, $runkB, $Nu, $Nuout) = SVM::read_merged_models($corpus.".".$COMMON::unkext.".".$modext.".".$dirext.".".$COMMON::MRGEXT, $omega, $verbose);

    my %M;
    $M{KNMODEL} = $KNMODEL;
    $M{UNKMODEL} = $UNKMODEL;
    $M{rB} = $rB;
    $M{runkB} = $runkB;
    $M{mode} = $mode;
    $M{fsk} = $fsk;
    $M{fsu} = $fsu;
   
    return \%M;
}

sub SVMT_load_models
{
   #description _ responsible for loading SVMT models.
   #param1  _ model directory
   #param2  _ direction mode (LR/RL/LRL)
   #param3  _ mode (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
   #               (                  3-unsupervised :: 4-unkown words on training)    
   #param4  _ EPSILON threshold for W features relevance  [KNOWN WORDS] 
   #param5  _ OMEGA threshold for W features relevance  [UNKNOWN WORDS] 
   #param6  _ VERBOSE 
   #param7  _ ambiguous pos list reference
   #param8  _ unknown pos list reference
   #param9  _ feature set list reference for KNOWN words
   #param10 _ feature set list reference for UNKNOWN words
   #@return _ SVMT models

   my $model = shift;
   my $direction = shift;
   my $mode = shift;
   my $epsilon = shift;
   my $omega = shift;
   my $verbose = shift;
   my $rambp = shift;
   my $runkp = shift;
   my $fsk = shift;
   my $fsu = shift;

   my %M;
   if (($direction eq $COMMON::lrmode) or ($direction eq $COMMON::lrlmode) or ($direction eq $COMMON::glrlmode)) { #LEFT-TO-RIGHT or both
      $M{LR} = load_models($model, $epsilon, $omega, $verbose, $COMMON::lrmode, $mode, $rambp, $runkp, $fsk, $fsu);
   }
   if (($direction eq $COMMON::rlmode) or ($direction eq $COMMON::lrlmode) or ($direction eq $COMMON::glrlmode)) { #LEFT-TO-RIGHT or both
      $M{RL} = load_models($model, $epsilon, $omega, $verbose, $COMMON::rlmode, $mode, $rambp, $runkp, $fsk, $fsu);
   }
   $M{direction} = $direction;

   return \%M;
}

sub SVMT_load_dict
{
   #description _ responsible for loading an SVMT dictionary.
   #param1 _ dictionary filename
   #param2 _ ambiguous pos list file
   #param3 _ unknown pos list file
   #param4 _ VERBOSE 
   #@return _ dictionary object

   my $dict = shift;
   my $fambp = shift;
   my $funkp = shift;
   my $verbose = shift;

   #dictionary generation
   if ($verbose > $COMMON::verbose1) { print STDERR "READING DICTIONARY <$dict>...\n"; }
   my $rdict = new DICTIONARY($dict, $fambp, $funkp);
   if ($verbose > $COMMON::verbose1) { print STDERR "[DONE]\n"; }

   return $rdict;
}

# ==========================================================================================
# ========================== TAGGING =======================================================
# ==========================================================================================


sub tag_sample
{
   #description _ returns the tag scoring the highest among the possible ones
   #param1  _ CLSF filename hash reference 
   #param2  _ dictionary object 
   #param3  _ word 
   #param4  _ verbosity level
   #@return _ (max_pos, max_score)

   my $rclsf = shift;
   my $rdict = shift;
   my $w = shift;
   my $verbose = shift;

   my $Cj = $COMMON::emptypos;
   my $max = $SVMTAGGER::MIN_VALUE;

   foreach my $Ci (keys %{$rclsf}) {
      #print "($Ci) $$rclsf->{$Ci} <?> $max)\n";
      if ($rclsf->{$Ci} > $max) { $max = $rclsf->{$Ci}; $Cj = $Ci; }
      elsif ($rclsf->{$Ci} == $max) {
         my $mftpos = $rdict->get_mft($w);
         if ($mftpos eq $Ci) { $Cj = $Ci; }
      }
   }

   if ($verbose > $COMMON::verbose2) {
      print STDERR "---------------------------------------------------------------------------------------------\n";
      #print STDERR Dumper($rclsf);
      foreach my $p (keys %{$rclsf}) {
         print STDERR "score[$w, $p] = $rclsf->{$p}\n";
      }
      print STDERR "---------------------------------------------------------------------------------------------\n";
   }

   return ($Cj, $max);
}

sub classify_sample_merged
{
   #description _ runs a given ambiguous sample, given a mapping, through the
   #              given models (assuming they've already been learned).
   #param1 _ AMBPOS (ambiguous POSs) hash reference              
   #param2 _ WORD POTSER's DICTIONARY list reference             
   #param3 _ SAMPLE attribute list reference                     
   #param4 _ biases hash reference                               
   #param5 _ weights MERGED MODEL (mapping + primal weights)     
   #param6 _ VERBOSE 
   #@return _ CLSF file hash reference                           

   my $rambpos = shift;
   my $potser = shift;
   my $sample = shift;
   my $rB = shift;
   my $rW = shift;
   my $verbose = shift;

   my %CLSF;    #predictions

   foreach my $pos (@{$potser}) {
      if ($verbose > $COMMON::verbose3) { print STDERR "---------------------------------------------------------------------------------------------\nPOS = $pos\n---------------------------------------------------------------------------------------------\n"; }

      if (exists($rambpos->{$pos})) { #check only possibly ambiguous POS
         $CLSF{$pos} = 0; #INITIALIZE PREDICTION
         foreach my $att (@{$sample}) {
            if (defined($rW->{$att})) {
               if (defined($rW->{$att}->{$pos})) {
		  if ($verbose > $COMMON::verbose3)
                     { print STDERR "W[$att, $pos] = $rW->{$att}->{$pos}\n"; }
		  my $w = $rW->{$att}->{$pos};
                  $CLSF{$pos} = $CLSF{$pos} + $w;
               }
            }
	 }
      }

      #TERMINATE PREDICTIONS
      if ($verbose > $COMMON::verbose3) {
         print STDERR "W[$pos] = $CLSF{$pos}\n";
         print STDERR "B[$pos] = $rB->{$pos}\n";
         print STDERR "score[$pos] = $CLSF{$pos} - $rB->{$pos} = ", $CLSF{$pos} - $rB->{$pos}, "\n";
         print STDERR "---------------------------------------------------------------------------------------------\n";
      }
      $CLSF{$pos} -= $rB->{$pos};
   }

   return \%CLSF;
}

sub pos_tag_sample {
    #description _ procedure to determine the part-of-speech given a context.
    #param1  _ sliding window object
    #param2  _ DICTIONARY object
    #param3  _ given word
    #param4  _ SENTENCE general INFO
    #param5  _ is the word unknown?
    #param6  _ all POS hash reference
    #param7  _ possible POS hash reference
    #param8  _ SVMT model (LR/RL)
    #param9  _ direction
    #param10 _ VERBOSE 
    #@return _ (part-of-speech, pred, rclsf, Ftime, Ctime)

    my $rwindow = shift;
    my $rdict = shift;
    my $w = shift;
    my $sinfo = shift;
    my $unknown = shift;
    my $all = shift;
    my $possible = shift;
    my $M = shift;
    my $direction = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $rattribw = $rwindow->prepare($direction); 
    my $Ftime1 = new Benchmark;
    my $rattribs = ATTGEN::generate_features($rattribw, $rdict, $sinfo, $M->{mode}, $unknown ? $M->{fsu} : $M->{fsk},$Unihan,$BS);# have modified  add $Unihan
    my $Ftime2 = new Benchmark;
    my $Ftime = Benchmark::timediff($Ftime2,$Ftime1);
    my $sample = SVM::write_mem_sample($rattribs);
    my $rclsf;

    if ($verbose > $COMMON::verbose2) {
       print STDERR "---------------------------------------------------------------------------------------------\nWORD = $w [", $unknown? "U" : "K" ,"]\n---------------------------------------------------------------------------------------------\n";
       $rattribw->print;
       print STDERR "---------------------------------------------------------------------------------------------\n";
    }

    my $Ctime1 = new Benchmark;
    if ($unknown) { $rclsf = classify_sample_merged($all, $possible, $sample, $M->{runkB}, $M->{UNKMODEL}, $verbose); }
    else { $rclsf = classify_sample_merged($all, $possible, $sample, $M->{rB}, $M->{KNMODEL}, $verbose); }
    my $Ctime2 = new Benchmark;
    my $Ctime = Benchmark::timediff($Ctime2,$Ctime1);
    my ($pos, $pred) = tag_sample($rclsf, $rdict, $w, $verbose);

    return ($pos, $pred, $rclsf, $Ftime, $Ctime);
}

sub analyze_sample
{
    #description _ procedure to determine the part-of-speech given a context.
    #param1  _ sliding window object
    #param2  _ WHOLE MODELS
    #param3  _ SENTENCE general INFO
    #param4  _ SVMT model (LR/RL) 1
    #param5  _ SVMT model (LR/RL) 2
    #param6  _ mode? boolean: 0 -> do_normal :: 1 -> do_special :: 2 -> do_viterbi
    #param7  _ direction
    #param8  _ VERBOSE 
    #@return _ (word, part-of-speech, pred, rclsf, Ftime, Ctime)

    my $rwindow = shift;
    my $M = shift;
    my $sinfo = shift;
    my $M1 = shift;
    my $M2 = shift;
    my $mode = shift;
    my $direction = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $pos;
    my $pred;
    my $rclsf;
    my $Ftime;
    my $Ctime;

    my $w = $rwindow->get_core_word();
    my $potser = $M->{dict}->get_real_potser($w);
    my $all;
    
    if ($M->{dict}->unknown_word($w)) { $all = $M->{rUP}; } #UNKNOWN WORD
    else {
       if ($M->{dict}->ambiguous_word($w)) { $all = $M->{rAP}; } #AMBIGUOUS WORD
       else { return ($w, $potser->[0], 0, {$potser->[0] => 0}); } #KNOWN and NOT AMBIGUOUS
    }

    if ($mode == $SVMTAGGER::Mspecial) {
       if (any_unknown($rwindow, $M->{dict})) {
          ($pos, $pred, $rclsf, $Ftime, $Ctime) = pos_tag_sample($rwindow, $M->{dict}, $w, $sinfo, $M->{dict}->unknown_word($w), $all, $potser, $M2, $direction, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
       else {
          ($pos, $pred, $rclsf, $Ftime, $Ctime) = pos_tag_sample($rwindow, $M->{dict}, $w, $sinfo, $M->{dict}->unknown_word($w), $all, $potser, $M1, $direction, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
    }
    else { #Mdefault
       ($pos, $pred, $rclsf, $Ftime, $Ctime) = pos_tag_sample($rwindow, $M->{dict}, $w, $sinfo, $M->{dict}->unknown_word($w), $all, $potser, $M1, $direction, $verbose,$Unihan,$BS);# have modified  add $Unihan
    }

    return ($w, $pos, $pred, $rclsf, $Ftime, $Ctime);
}

sub process_sample_merged {
    #description _ procedure to determine the part-of-speech given a context.
    #param1  _ sliding window object
    #param2  _ given word
    #param3  _ given word pos (if available --> 2 passes)
    #param4  _ WHOLE MODELS
    #param5  _ SVMT model (LR/RL) 1
    #param6  _ SVMT model (LR/RL) 2
    #param7  _ SENTENCE general INFO
    #param8  _ mode? boolean: 0 -> do_normal :: 1 -> do_special :: 2 -> do_viterbi
    #param9  _ direction
    #param10 _ whole element entry (list), e.g {WORD PoS IOB ...}
    #param11 _ VERBOSE 
    #@return _ (word, part-of-speech, pred, rclsf, Ftime, Ctime)

    my $rwindow = shift;
    my $word = shift;
    my $wordpos = shift;
    my $M = shift;
    my $M1 = shift;
    my $M2 = shift;
    my $sinfo = shift;
    my $mode = shift;
    my $direction = shift;
    my $entry = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $w;
    my $pos;
    my $pred;
    my $rclsf;
    my $Ftime;
    my $Ctime;

    #shift sliding window one position left
    $rwindow->lshift(1);    

    #push current entry onto the sliding window
    if ($word ne "") {
       if ($wordpos ne "") { $rwindow->push($word, $wordpos, $rwindow->get_len - 1, $entry); }
       else { $rwindow->push($word, $COMMON::emptypos, $rwindow->get_len - 1, $entry); }
    }

    #is the core item active?
    if ($rwindow->active()) {
       ($w, $pos, $pred, $rclsf, $Ftime, $Ctime) = analyze_sample($rwindow, $M, $sinfo, $M1, $M2, $mode, $direction, $verbose,$Unihan,$BS);# have modified  add $Unihan
       # --> feed back always
       $rwindow->push($w, $pos, $rwindow->get_core);
    }

    
    return ($w, $pos, $pred, $rclsf, $Ftime, $Ctime);
}

sub do_greedy_tagging
{
    #description _ responsible for tagging in one-pass [one-pass LR/RL]
    #param1  _ WHOLE MODELS
    #param2  _ VERBOSE 
    #param3  _ text direction "LR" for left-to-right ; "RL" for right-to-left;
    #param4  _ input list reference
    #param5  _ SVMT model (LR/RL) 1
    #param6  _ SVMT model (LR/RL) 2
    #param7  _ mode? boolean: 0 -> do_normal :: 1 -> do_special :: 2 -> do_viterbi
    #@return _ (output LR list reference, global score, Feature-time, Computation-time-svm)

    my $M = shift;
    my $verbose = shift;
    my $direction = shift;
    my $input_LR = shift;
    my $M1 = shift;
    my $M2 = shift;
    my $mode = shift;
    my $Unihan = shift;
    my $BS = shift;

    my @OUT;
    my $SIZE = scalar(@{$input_LR});
    my $Ftime;
    my $Ctime;

    my $iter = 0;
    my $sinfo;
    my $word;
    my $wordpos;
    my $rwindow = new SWINDOW($M->{WS}->[0], $M->{WS}->[1]); #sliding window generation

    if ($direction eq $COMMON::lrmode) {
       $sinfo = get_sentence_info_list_entry($input_LR, $COMMON::lrmode, 0);
    }
    elsif ($direction eq $COMMON::rlmode) {
       $sinfo = get_sentence_info_list_entry($input_LR, $COMMON::rlmode, $SIZE - 1);
    }

    if ($verbose > $COMMON::verbose2) {
       if ($direction eq $COMMON::lrmode) { print STDERR $iter; }
       elsif ($direction eq $COMMON::rlmode) { print STDERR $SIZE; }
    }       

    while ($iter < $SIZE) {
       my $columns;
       if ($direction eq $COMMON::lrmode) {
          $word = $input_LR->[$iter]->get_word;
          $wordpos = $input_LR->[$iter]->get_pos;
          if (COMMON::end_of_sentence($word)) {
             $sinfo = get_sentence_info_list_entry($input_LR, $direction, $iter);
          }
          $columns = $input_LR->[$iter]->get_cols;
       }
       elsif ($direction eq $COMMON::rlmode) {
          $word = $input_LR->[$SIZE - $iter - 1]->get_word;
          $wordpos = $input_LR->[$SIZE - $iter - 1]->get_pos;
          if (COMMON::end_of_sentence($word)) {
             $sinfo = get_sentence_info_list_entry($input_LR, $direction, $SIZE - $iter - 1);
          }
          $columns = $input_LR->[$SIZE - $iter - 1]->get_cols;
       }
       if ($word ne "") { # don't process emtpy lines
          shift(@{$columns});
          my ($w, $pos, $pred, $rclsf, $SFtime, $SCtime) = process_sample_merged($rwindow, $word, $wordpos, $M, $M1, $M2, $sinfo, $mode, $direction, $columns, $verbose,$Unihan,$BS);# have modified  add $Unihan
          if (defined($SFtime)) { $Ftime += $SFtime->[1]; }
          if (defined($SCtime)) { $Ctime += $SCtime->[1]; }
          if ($w ne "") { # empty words don't get tagged
             my $rOUT = new ENTRY($w, $pos, $rclsf, $columns);
             if ($direction eq $COMMON::lrmode) { push(@OUT, $rOUT); }
             elsif ($direction eq $COMMON::rlmode) { unshift(@OUT, $rOUT); }
          }
       }
       $iter++;
       if ($verbose > $COMMON::verbose2) {
          if ($direction eq $COMMON::lrmode) { COMMON::show_progress($iter, $COMMON::progress1, $COMMON::progress2); }
          elsif ($direction eq $COMMON::rlmode) { COMMON::show_progress($SIZE - $iter - 1, $COMMON::progress1, $COMMON::progress2); }       
       }
    }

    # ------------------------------------ last words -------------------------------------------
    my $i = $M->{WS}->[0] - $M->{WS}->[1] - 1;
    while ($i > 0) { #process last words in the sliding window
       my ($w, $pos, $pred, $rclsf, $SFtime, $SCtime) = process_sample_merged($rwindow, $COMMON::emptyword, $COMMON::emptypos, $M, $M1, $M2, $sinfo, $mode, $direction, 0, $verbose,$Unihan,$BS);# have modified  add $Unihan
       if (defined($SFtime)) { $Ftime += $SFtime->[1]; }
       if (defined($SCtime)) { $Ctime += $SCtime->[1]; }
       if ($w ne "") { # empty words don't get tagged
          my $rOUT = new ENTRY($w, $pos, $rclsf, 0);
          if ($direction eq $COMMON::lrmode) { push(@OUT, $rOUT); }
          elsif ($direction eq $COMMON::rlmode) { unshift(@OUT, $rOUT); }
       }
       $i--;
    }

    if ($verbose > $COMMON::verbose2) {
       if ($direction eq $COMMON::lrmode) { print STDERR ".$iter [DONE]\n"; }
       elsif ($direction eq $COMMON::rlmode) { print STDERR " [DONE]\n"; }
    }       

    return(\@OUT, 0, $Ftime, $Ctime);
}

sub build_window
{
    #description _ responsible for constructing a new window
    #param1  _ input sentence
    #param2  _ core index
    #param3  _ window setup
    #param4  _ DICTIONARY object
    #@return _ sliding window

    my $sentence = shift;
    my $core = shift;
    my $winsetup = shift;
    my $rdict = shift;

    my $SIZE = scalar(@{$sentence});
    my $rwindow = new SWINDOW($winsetup->[0], $winsetup->[1]); #sliding window generation
    my $i = 0;
    while ($i < $winsetup->[0]) {
       my $current;
       $current = $core + $i - ($winsetup->[1]);
       if (($current >= 0) and ($current < $SIZE)) {
          $rwindow->push($sentence->[$current], $COMMON::emptypos, $i);
          my $potser = $rdict->get_real_potser($rwindow->get_word($i));
          #my $potser = $rdict->get_real_potser($rwindow->get_actual_word($i));
          $rwindow->set_kamb($i, $potser);
       }
       $i++;
    }

    return $rwindow;
}

sub apply_softmax {
   #description _ apply softmax function over a set of predictions.
   #param1  _ prediction hash reference
   #param2  _ softmax mode 0: do nothing
   #        _              1: do ln(e^score(i) / [sum:1<=j<=N:[e^score(j)]])

   my $rclsf = shift;
   my $mode = shift;
   
   if ($mode > 0)  { #APPLY SOFTMAX?
      my $Psum = 0;
      foreach my $Ci (keys %{$rclsf}) { $Psum += exp($rclsf->{$Ci}); }
      foreach my $Ci (keys %{$rclsf}) { $rclsf->{$Ci} = log(exp($rclsf->{$Ci}) / $Psum); }
   }

   return $rclsf;
}

sub path_beam_cutoff
{
    #description _ beam cutoff procedure
    #param1 _ path structure element
    #param2 _ beam count cutoff
    #param3 _ beam ratio cutoff

    my $elem = shift;
    my $nbeams = shift;
    my $bratio = shift;

    my @l; #list of predictions
    foreach my $k (keys %{$elem->[0]}) { push(@l, $elem->[0]->{$k}); }
    my @lsorted = sort {$a <=> $b} @l; #sorted list of predictions

    #nbeam cutoff
    if (($nbeams >= 1) and (scalar(@lsorted) > $nbeams)) { #select beams over cutoff point
       my $cutoff = $lsorted[scalar(@lsorted) - $nbeams];
       foreach my $p (keys %{$elem->[0]}) {
	  if ($elem->[0]->{$p} < $cutoff) { delete $elem->[0]->{$p}; delete $elem->[1]->{$p} }
       }
    }

    #beam ratio cutoff
    if ($bratio != 0) { #select beams over cutoff point
       my $cutoff = $lsorted[scalar(@lsorted) - 1];
       $cutoff = ($cutoff >= 0)? $cutoff * $bratio : $cutoff /$bratio;
       foreach my $p (keys %{$elem->[0]}) {
	  if ($elem->[0]->{$p} < $cutoff) { delete $elem->[0]->{$p}; delete $elem->[1]->{$p} }
       }
    }
}

sub build_left_context
{
    #description _ builds left window context
    #param1 _ window
    #param2 _ path
    #param3 _ path position
    #param4 _ current PoS

    my $rwindow = shift;
    my $path = shift;
    my $i = shift;
    my $pos = shift;

    my $core = $rwindow->get_core;
    my $len = $rwindow->get_len;

    $rwindow->set_pos($core - 1, $pos);

    my $j = 1;
    while (($j < ($len - $core + 1)) and ($i > 0) and (!($rwindow->is_empty($core - $j - 1)))) {
       $pos = $path->[$i][1]{$pos};
       $rwindow->set_pos($core - $j - 1, $pos);
       $j++;
       $i--;
    }
}

sub compute_tagging
{
    #description _ computes the tagging
    #param1 _ window
    #param2 _ whole models
    #param3 _ SENTENCE general INFO
    #param4 _ DICTIONARY object
    #param5 _ direction (LR/RL)
    #param6  _ ambiguous pos hash reference
    #param7  _ unknown pos hash reference
    #param8 _ verbose

    my $rwindow = shift;
    my $M = shift;
    my $sinfo = shift;
    my $rdict = shift;
    my $direction = shift;
    my $rambp = shift;
    my $runkp = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $word = $rwindow->get_core_word();
    #my $word = $rwindow->get_core_actual_word();
    my $possible = $rdict->get_real_potser($word);
    my $unknown = $rdict->unknown_word($word);
    my $rB;
    my $MODEL;
    my $rall;
    if ($unknown) { $rB = $M->{runkB}; $MODEL = $M->{UNKMODEL}; $rall = $runkp; } #UNKNOWN WORD
    else { $rB = $M->{rB}; $MODEL = $M->{KNMODEL}; $rall = $rambp; } #AMBIGUOUS KNOWN WORD

    #generate features
    my $rattribw = $rwindow->prepare($direction); 
    if ($verbose > $COMMON::verbose2) {
       print STDERR "---- compute tagging scores -----------------------------------------------------------------\nWORD = $word [", $unknown? "U" : "K" ,"]\n---------------------------------------------------------------------------------------------\n";
       $rattribw->print;
       print STDERR "---------------------------------------------------------------------------------------------\n";
    }
    my $Ftime1 = new Benchmark;
    my $rattribs = ATTGEN::generate_features($rattribw, $rdict, $sinfo, $M->{mode}, $unknown ? $M->{fsu} : $M->{fsk},$Unihan,$BS);# have modified  add $Unihan
    my $Ftime2 = new Benchmark;
    my $Ftimediff = Benchmark::timediff($Ftime2,$Ftime1);
    my $Ftime = $Ftimediff->[1];
    my $sample = SVM::write_mem_sample($rattribs);

    #get_SVM_score
    my $Ctime1 = new Benchmark;
    my $rclsf = classify_sample_merged($rall, $possible, $sample, $rB, $MODEL, $verbose);
    my $Ctime2 = new Benchmark;
    my $Ctimediff = Benchmark::timediff($Ctime2,$Ctime1);
    my $Ctime = $Ctimediff->[1];

    if ($verbose > $COMMON::verbose2) {
       foreach my $k (keys %{$rclsf}) {
          print STDERR "score($k) = ", $rclsf->{$k}, "\n";
       }
    }

    return($rclsf, $Ftime, $Ctime);
}

sub compute_scores_old {
    #description _ viterbi sentence-level PoS-tagging (induction step)
    #param1  _ sliding window object
    #param2  _ DICTIONARY object
    #param3  _ SENTENCE general INFO
    #param4  _ SVMT model (LR/RL)
    #param5  _ path structure
    #param6  _ current sentence position index
    #param7  _ direction (LR/RL)
    #param8  _ ambiguous pos hash reference
    #param9  _ unknown pos hash reference
    #param10 _ beam count cutoff
    #param11 _ beam search ratio
    #param12 _ softmax mode 0: do nothing
    #        _              1: do ln(e^score(i) / [sum:1<=j<=N:[e^score(j)]])
    #param13 _ VERBOSE 
    #@return _ (part-of-speech, global pred, local rclsf, Ftime, Ctime)

    my $rwindow = shift;
    my $rdict = shift;
    my $sinfo = shift;
    my $M = shift;
    my $path = shift;
    my $i = shift;
    my $direction = shift;
    my $rambp = shift;
    my $runkp = shift;
    my $nbeams = shift;
    my $bratio = shift;
    my $softmax = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $word = $rwindow->get_core_word();
    my $possible = $rdict->get_real_potser($word);

    my $Ftime = 0;  #Feature extraction time (FEX)
    my $Ctime = 0;  #Calculation time (SVM)
    my %CLSFG;      #global predictions

    my %maxpred;
    my %maxGpred;
    my %maxGpos;
    foreach my $t (@{$possible}) {
       $maxpred{$t} = $SVMTAGGER::MIN_VALUE;
       $maxGpred{$t} = $SVMTAGGER::MIN_VALUE;
       $maxGpos{$t} = $COMMON::emptypos;
    }
    foreach my $tj (@{$possible}) {
       #print STDERR "------------------------------ i = $i :: tj = $tj ---------------------------------\n";
       my $max = $SVMTAGGER::MIN_VALUE;
       if ($i == 0) { # no left context is available
          my ($rclsf, $ftime, $ctime) = compute_tagging($rwindow, $M, $sinfo, $rdict, $direction, $rambp, $runkp, $verbose,$Unihan,$BS);# have modified  add $Unihan
          #APPLY softmax
          apply_softmax($rclsf, $softmax);
          $Ftime += $ftime; $Ctime += $ctime;
          foreach my $k (keys %{$rclsf}) {
             if ($rclsf->{$k} >= $maxGpred{$k}) { $maxGpred{$k} = $rclsf->{$k}; $maxpred{$k} = $rclsf->{$k}; }
	  }
       }
       else { # left context must be build according to the current path
          foreach my $tk (keys %{$path->[$i]->[0]}) {
             build_left_context($rwindow, $path, $i, $tk);
    	     #print STDERR "****** tk = $tk ******\n";
             my ($rclsf, $ftime, $ctime) = compute_tagging($rwindow, $M, $sinfo, $rdict, $direction, $rambp, $runkp, $verbose,$Unihan,$BS);# have modified  add $Unihan
             #APPLY softmax
             apply_softmax($rclsf, $softmax);
             $Ftime += $ftime; $Ctime += $ctime;
             if ($softmax) { #PRODUCT
                foreach my $k (keys %{$rclsf}) {
	  	   if (($rclsf->{$k} * $path->[$i]->[0]->{$tk}) >= $maxGpred{$k}) {
                      $maxGpred{$k} = $rclsf->{$k} * $path->[$i]->[0]->{$tk};
                      $maxpred{$k} = $rclsf->{$k};
	           }
  	        }
                if (($rclsf->{$tj} * $path->[$i]->[0]->{$tk}) >= $max)
                   { $max = $rclsf->{$tj} * $path->[$i]->[0]->{$tk}; $maxGpos{$tj} = $tk; }
	     }
             else { #ADD
                foreach my $k (keys %{$rclsf}) {
	  	   if (($rclsf->{$k} + $path->[$i]->[0]->{$tk}) >= $maxGpred{$k}) {
                      $maxGpred{$k} = $rclsf->{$k} + $path->[$i]->[0]->{$tk};
                      $maxpred{$k} = $rclsf->{$k};
	           }
  	        }
                if (($rclsf->{$tj} + $path->[$i]->[0]->{$tk}) >= $max)
                   { $max = $rclsf->{$tj} + $path->[$i]->[0]->{$tk}; $maxGpos{$tj} = $tk; }
	     }
          }
       }
       $path->[$i+1]->[0]->{$tj} = $maxGpred{$tj};
       $path->[$i+1]->[1]->{$tj} = $maxGpos{$tj};
       $path->[$i+1]->[2]->{$tj} = $maxpred{$tj};
       if ($verbose > $COMMON::verbose2) {
	  print STDERR "---------------------------------------\n";
	  print STDERR "score(", $i+1, ", ", $tj, ") = ", $path->[$i+1]->[2]->{$tj}, "\n";
	  print STDERR "d(", $i+1, ", ", $tj, ") = ", $path->[$i+1]->[0]->{$tj}, "\n";
	  print STDERR "f(", $i+1, ", ", $tj, ") = ", $path->[$i+1]->[1]->{$tj}, "\n";
       }
    }

    if (($nbeams > 0) or ($bratio != 0)) { path_beam_cutoff($path->[$i+1], $nbeams, $bratio); }
    
    return ($Ftime, $Ctime);
}

sub compute_scores {
    #description _ viterbi sentence-level PoS-tagging (induction step)
    #param1  _ sliding window object
    #param2  _ DICTIONARY object
    #param3  _ SENTENCE general INFO
    #param4  _ SVMT model (LR/RL)
    #param5  _ path structure
    #param6  _ current sentence position index
    #param7  _ direction (LR/RL)
    #param8  _ ambiguous pos hash reference
    #param9  _ unknown pos hash reference
    #param10 _ beam count cutoff
    #param11 _ beam search ratio
    #param12 _ softmax mode 0: do nothing
    #        _              1: do ln(e^score(i) / [sum:1<=j<=N:[e^score(j)]])
    #param13 _ VERBOSE 
    #@return _ (part-of-speech, global pred, local rclsf, Ftime, Ctime)

    my $rwindow = shift;
    my $rdict = shift;
    my $sinfo = shift;
    my $M = shift;
    my $path = shift;
    my $i = shift;
    my $direction = shift;
    my $rambp = shift;
    my $runkp = shift;
    my $nbeams = shift;
    my $bratio = shift;
    my $softmax = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $word = $rwindow->get_core_word();
    my $possible = $rdict->get_real_potser($word);

    my $Ftime = 0;  #Feature extraction time (FEX)
    my $Ctime = 0;  #Calculation time (SVM)
    my %CLSFG;      #global predictions

    my %maxpred;
    my %maxGpred;
    my %maxGpos;
    foreach my $t (@{$possible}) {
       $maxpred{$t} = $SVMTAGGER::MIN_VALUE;
       $maxGpred{$t} = $SVMTAGGER::MIN_VALUE;
       $maxGpos{$t} = $COMMON::emptypos;
    }

    foreach my $tk (keys %{$path->[$i]->[0]}) {
       my $max = $SVMTAGGER::MIN_VALUE;
       if ($i == 0) { # no left context is available
          my ($rclsf, $ftime, $ctime) = compute_tagging($rwindow, $M, $sinfo, $rdict, $direction, $rambp, $runkp, $verbose,$Unihan,$BS);# have modified  add $Unihan
          apply_softmax($rclsf, $softmax);
          $Ftime += $ftime; $Ctime += $ctime;
          foreach my $k (keys %{$rclsf}) {
             if ($rclsf->{$k} >= $maxGpred{$k}) { $maxGpred{$k} = $rclsf->{$k}; $maxpred{$k} = $rclsf->{$k}; }
	  }
       }
       else { # left context must be build according to the current path
          build_left_context($rwindow, $path, $i, $tk);
          my ($rclsf, $ftime, $ctime) = compute_tagging($rwindow, $M, $sinfo, $rdict, $direction, $rambp, $runkp, $verbose,$Unihan,$BS);# have modified  add $Unihan
          $Ftime += $ftime; $Ctime += $ctime;
          foreach my $tj (@{$possible}) {
             apply_softmax($rclsf, $softmax);
             foreach my $k (keys %{$rclsf}) {
		if (($rclsf->{$k} + $path->[$i]->[0]->{$tk}) >= $maxGpred{$k}) {
                   $maxGpred{$k} = $rclsf->{$k} + $path->[$i]->[0]->{$tk};
                   $maxpred{$k} = $rclsf->{$k};
                   $maxGpos{$k} = $tk;
	        }
  	     }
	  }
       }
    }

    foreach my $tj (@{$possible}) {
       $path->[$i+1]->[0]->{$tj} = $maxGpred{$tj}; #d(i+1, $tj)
       $path->[$i+1]->[1]->{$tj} = $maxGpos{$tj};  #f(i+1, $tj)
       $path->[$i+1]->[2]->{$tj} = $maxpred{$tj};
       if ($verbose > $COMMON::verbose2) {
	  print STDERR "---------------------------------------\n";
	  print STDERR "score(", $i+1, ", ", $tj, ") = ", $path->[$i+1]->[2]->{$tj}, "\n";
	  print STDERR "d(", $i+1, ", ", $tj, ") = ", $path->[$i+1]->[0]->{$tj}, "\n";
	  print STDERR "f(", $i+1, ", ", $tj, ") = ", $path->[$i+1]->[1]->{$tj}, "\n";
       }
    }

    if (($nbeams > 0) or ($bratio != 0)) { path_beam_cutoff($path->[$i+1], $nbeams, $bratio); }
    
    return ($Ftime, $Ctime);
}

sub Hmax
{
    #description _ returns the (key, value) pair of the max value found in a hash structure.
    #param1 _ hash structure
    #@return _ (maxkey, maxvalue)

    my $hash = shift;

    my $maxk;
    my $maxv = $SVMTAGGER::MIN_VALUE;
    foreach my $k (keys %{$hash}) {
       if ($hash->{$k} > $maxv) { $maxv = $hash->{$k}; $maxk = $k; }
    }

    return ($maxk, $maxv);
}

sub do_viterbi_sentence
{
    #description _ responsible for performing viterbi tagging
    #param1  _ WHOLE MODELS
    #param2  _ text direction "LR" for left-to-right ; "RL" for right-to-left; "LRL" for both
    #param3  _ input LR sentence list reference
    #param4  _ SVMT models (LR/RL)
    #param5  _ VERBOSE 
    #@return _ (output LR sentence list reference, global score, Ftime, Ctime)

    my $M = shift;
    my $direction = shift;
    my $input_LR = shift;
    my $VM = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    #INITIALIZATION
    my @OUT;
    my @PATH;
    my $Ctime = 0; # (SVM)
    my $Ftime = 0; # (FEX)
    my $SIZE = scalar(@{$input_LR});
    my $sinfo = get_sentence_info_list($input_LR, $COMMON::lrmode, scalar(@{$input_LR}) - 1);
    my $input;
    if ($direction eq $COMMON::lrmode) { $input = $input_LR; }
    elsif ($direction eq $COMMON::rlmode) { $input = COMMON::reverse_list($input_LR); }
    $PATH[0] = [{ "." => 0 }, { "." => $COMMON::emptypos }, { "." => 0 }];

    #INDUCTION
    my $i = 0;
    while ($i < $SIZE) {
       my $rwindow = build_window($input, $i, $M->{WS}, $M->{dict});
       my $word = $rwindow->get_core_word();

       if (($M->{dict}->unknown_word($word)) or ($M->{dict}->ambiguous_word($word))) {
          # word is either unknown or known_ambiguous
          my ($SFtime, $SCtime) = compute_scores($rwindow, $M->{dict}, $sinfo, $VM, \@PATH, $i, $direction, $M->{rAP}, $M->{rUP}, $M->{nbeams}, $M->{bratio}, $M->{softmax}, $verbose,$Unihan,$BS);# have modified  add $Unihan
          $Ftime += $SFtime; $Ctime += $SCtime;
       }
       else { # word is known_unambiguous
	  my $pos = $M->{dict}->get_mft($word);
          my ($maxk, $maxv) = Hmax($PATH[$i][0]);
	  $PATH[$i+1]->[0]->{$pos} = $maxv;
          $PATH[$i+1]->[1]->{$pos} = $maxk;
          $PATH[$i+1]->[2]->{$pos} = 0;
       }

       $i++;
    }

    #EXAMINE FINAL PATH
    if ($verbose > $COMMON::verbose2) {
       print STDERR "---------------------------------------\n";
       print STDERR "viterbi path structure\n";
       my $j = 0;
       while ($j < scalar(@PATH)) {
	  print STDERR "------------------ $j -----------------\n";
	  foreach my $k (keys %{$PATH[$j]->[0]}) {
	     print STDERR "d($j, $k) = ", $PATH[$j][0]->{$k}, "\n";
	     print STDERR "f($j, $k) = ", $PATH[$j][1]->{$k}, "\n";
	     print STDERR "s($j, $k) = ", $PATH[$j][2]->{$k}, "\n";
	  }
          $j++;
       }
       print STDERR "---------------------------------------\n";
    }

    #TERMINATION -> path readout
    $i = $SIZE - 1;
    my ($pos, $Gscore) = Hmax($PATH[$i+1]->[0]);
    while ($i >= 0) {
       my $rOUT = new ENTRY($input->[$i], $pos, $PATH[$i+1]->[2], 0);
       if ($direction eq $COMMON::lrmode) { unshift(@OUT, $rOUT); }
       else { push(@OUT, $rOUT); }
       $pos = $PATH[$i+1]->[1]{$pos};
       $i--;
    }

    return(\@OUT, $Gscore, $Ftime, $Ctime);
}

sub push_sentence
{
    #description _ it appends a sentence list to another
    #param1  _ target list
    #param2  _ source list

    my $target = shift;
    my $source = shift;

    my $i;
    while ($i < scalar(@{$source})) {
       push(@{$target}, $source->[$i]);
       $i++;
    }

}

sub do_viterbi_tagging
{
    #description _ responsible for performing tagging one-pass [one-pass + LRL]
    #param1  _ WHOLE MODELS
    #param2  _ VERBOSE 
    #param3  _ text direction "LR" for left-to-right ; "RL" for right-to-left; "LRL" for both
    #param4  _ input LR list reference
    #param5  _ SVMT models
    #@return _ (output LR list reference, global score, Ftime, Ctime)

    my $M = shift;
    my $verbose = shift;
    my $direction = shift;
    my $input_LR = shift;
    my $VM = shift;
    my $Unihan = shift;
    my $BS = shift;

    my @OUT;
    my @SENTENCE;
    my $Ftime = 0;  # (FEX)
    my $Ctime = 0;  # (SVM)
    my $Gscore = 0;

    my $iter = 0;
    my $SIZE = scalar(@{$input_LR});

    if ($verbose > $COMMON::verbose2) {
       if ($direction eq $COMMON::lrmode) { print STDERR $iter; }
       elsif ($direction eq $COMMON::rlmode) { print STDERR $SIZE; }
    }       

    while ($iter < $SIZE) {
       my $word = $input_LR->[$iter]->get_word;
       if ($word ne "") { # don't process emtpy lines
          push(@SENTENCE, $word);
          if ((COMMON::end_of_sentence($word)) or ($iter == $SIZE - 1)) {
	     if (!(COMMON::end_of_sentence($word))) { push(@SENTENCE, "¡£"); }#modify "."
             my ($out, $auxGscore, $SFtime, $SCtime) = do_viterbi_sentence($M, $direction, \@SENTENCE, $VM, $verbose,$Unihan,$BS);# have modified  add $Unihan
             $Ftime += $SFtime; $Ctime += $SCtime; $Gscore += $auxGscore;
             push_sentence(\@OUT, $out);
             @SENTENCE = ();
          }
       }
       $iter++;
       if ($verbose > $COMMON::verbose2) {
          if ($direction eq $COMMON::lrmode) { COMMON::show_progress($iter, $COMMON::progress1, $COMMON::progress2); }
          elsif ($direction eq $COMMON::rlmode) { COMMON::show_progress($SIZE - $iter - 1, $COMMON::progress1, $COMMON::progress2); }       
       }
    }

    if ($verbose > $COMMON::verbose2) {
       if ($direction eq $COMMON::lrmode) { print STDERR ".$iter [DONE]\n"; }
       elsif ($direction eq $COMMON::rlmode) { print STDERR " [DONE]\n"; }
    }       

    return(\@OUT, $Gscore, $Ftime, $Ctime);
}

sub do_tagging
{
    #description _ responsible for tagging in one-pass [one-pass LR/RL]
    #param1  _ WHOLE MODELS
    #param2  _ VERBOSE 
    #param3  _ text direction "LR" for left-to-right ; "RL" for right-to-left;
    #param4  _ input LR list reference
    #param5  _ SVMT model (LR/RL) 1
    #param6  _ SVMT model (LR/RL) 2
    #param7  _ mode? boolean: 0 -> do_normal :: 1 -> do_special :: 2 -> do_viterbi
    #@return _ (output LR list reference, global score, Feature-time, Computation-time-svm)

    my $M = shift;
    my $verbose = shift;
    my $direction = shift;
    my $input_LR = shift;
    my $M1 = shift;
    my $M2 = shift;
    my $mode = shift;
    my $Unihan = shift;
    my $BS = shift;

    my @OUT;

    if ($verbose > $COMMON::verbose2) {
       print STDERR "TAGGING < DIRECTION = ";
       if ($direction eq $COMMON::lrmode) { print STDERR "left-to-right"; }
       elsif ($direction eq $COMMON::rlmode) { print STDERR "right-to-left"; }
       elsif ($direction eq $COMMON::lrlmode) { print STDERR "left-to-right then right-to-left"; }
       elsif ($direction eq $COMMON::glrlmode) { print STDERR "left-to-right then right-to-left (global)"; }
       else { print STDERR "$direction"; }
       print STDERR " :: MODE = ";
       if ($mode == $SVMTAGGER::Mdefault) { print STDERR "normal"; }
       elsif ($mode == $SVMTAGGER::Mspecial) { print STDERR "special"; }
       elsif ($mode == $SVMTAGGER::Mviterbi) { print STDERR "viterbi"; }
       else { print STDERR "$mode"; }
       print STDERR " :: MODEL = ";
       if ($M1->{mode} == $COMMON::mode0) { print STDERR "ambiguous context"; }
       elsif ($M1->{mode} == $COMMON::mode1) { print STDERR "disambiguated context"; }
       elsif ($M1->{mode} == $COMMON::mode2) { print STDERR "no context"; }
       else { print STDERR $M1->{mode}; }
       if ($M2 != 0) {
          print STDERR " :: MODEL2 = ";
          if ($M2->{mode} == $COMMON::mode0) { print STDERR "ambiguous context"; }
          elsif ($M2->{mode} == $COMMON::mode1) { print STDERR "disambiguated context"; }
          elsif ($M2->{mode} == $COMMON::mode2) { print STDERR "no context"; }
          else { print STDERR $M2->{mode}; }
       }
       print STDERR " >\n";
    }

    if ($mode == $SVMTAGGER::Mviterbi) {
       return do_viterbi_tagging($M, $verbose, $direction, $input_LR, $M1,$Unihan,$BS);# have modified  add $Unihan
    }
    else {  #Mdefault or Mspecial
       return do_greedy_tagging($M, $verbose, $direction, $input_LR, $M1, $M2, $mode,$Unihan,$BS);# have modified  add $Unihan
    }
}

sub do_tagging_LRL_1P
{
    #description _ responsible for performing tagging one-pass [one-pass + LRL]
    #param1  _ WHOLE MODELS
    #param2  _ VERBOSE 
    #param3  _ text direction "LR" for left-to-right ; "RL" for right-to-left; "LRL" for both
    #param4  _ input list reference
    #param5  _ SVMT models 1
    #param6  _ SVMT models 2
    #param7  _ mode: 0 -> do_normal :: 1 -> do_special :: 2 -> do_viterbi
    #@return _ (output list reference, Ftime, Ctime)

    my $M = shift;
    my $verbose = shift;
    my $direction = shift;
    my $input_LR = shift;
    my $M1 = shift;
    my $M2 = shift;
    my $mode = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $OUT_LR;
    my $OUT_RL;
    my $GscoreLR;
    my $GscoreRL;
    my $Ftime;
    my $Ctime;
    my $FtimeLR;
    my $CtimeLR;
    my $FtimeRL;
    my $CtimeRL;

    # -----> TAGGING ------------------------------------------------------------------------------

    if ($direction eq $COMMON::lrmode) {      # LEFT-TO-RIGHT
       if ($M2 == 0) {
          ($OUT_LR, $GscoreLR, $Ftime, $Ctime) = do_tagging($M, $verbose, $COMMON::lrmode, $input_LR, $M1->{LR}, 0, $mode,$Unihan,$BS);# have modified  add $Unihan
       }
       else {
          ($OUT_LR, $GscoreLR, $Ftime, $Ctime) = do_tagging($M, $verbose, $COMMON::lrmode, $input_LR, $M1->{LR}, $M2->{LR}, $mode,$Unihan,$BS);# have modified  add $Unihan
       }
       return($OUT_LR, $Ftime, $Ctime);
    }
    elsif ($direction eq $COMMON::rlmode) {      # RIGHT-TO-LEFT
       if ($M2 == 0) {
          ($OUT_RL, $GscoreRL, $Ftime, $Ctime) = do_tagging($M, $verbose, $COMMON::rlmode, $input_LR, $M1->{RL}, 0, $mode,$Unihan,$BS);# have modified  add $Unihan
       }
       else {
          ($OUT_RL, $GscoreRL, $Ftime, $Ctime) = do_tagging($M, $verbose, $COMMON::rlmode, $input_LR, $M1->{RL}, $M2->{RL}, $mode,$Unihan,$BS);# have modified  add $Unihan
       }
       return($OUT_RL, $Ftime, $Ctime);
    }
    elsif ($direction eq $COMMON::lrlmode) {      #both LEFT-TO-RIGHT and RIGHT-TO-LEFT
       if ($M2 == 0) {
          ($OUT_LR, $GscoreLR, $FtimeLR, $CtimeLR) = do_tagging($M, $verbose, $COMMON::lrmode, $input_LR, $M1->{LR}, 0, $mode,$Unihan,$BS);# have modified  add $Unihan
          ($OUT_RL, $GscoreRL, $FtimeRL, $CtimeRL) = do_tagging($M, $verbose, $COMMON::rlmode, $input_LR, $M1->{RL}, 0, $mode,$Unihan,$BS);# have modified  add $Unihan
          $Ftime = $FtimeLR + $FtimeRL;
          $Ctime = $CtimeLR + $CtimeRL;
       }
       else {
          ($OUT_LR, $GscoreLR, $FtimeLR, $CtimeLR) = do_tagging($M, $verbose, $COMMON::lrmode, $input_LR, $M1->{LR}, $M2->{LR}, $mode,$Unihan,$BS);# have modified  add $Unihan
          ($OUT_RL, $GscoreRL, $FtimeRL, $CtimeRL) = do_tagging($M, $verbose, $COMMON::rlmode, $input_LR, $M1->{RL}, $M2->{RL}, $mode,$Unihan,$BS);# have modified  add $Unihan
          $Ftime = $FtimeLR + $FtimeRL;
          $Ctime = $CtimeLR + $CtimeRL;
       }
       if ($verbose > $COMMON::verbose2) { print STDERR "COMBINING LR/RL...0"; }
       my $iter = 0;
       my $SIZE = scalar(@{$input_LR});
       my @OUT_LRL;
       while ($iter < $SIZE) {
	  apply_softmax($OUT_LR->[$iter]->get_pp, $M->{softmax});
	  apply_softmax($OUT_RL->[$iter]->get_pp, $M->{softmax});
          if ($OUT_LR->[$iter]->get_pred >= $OUT_RL->[$iter]->get_pred) { push(@OUT_LRL, $OUT_LR->[$iter]); }
          else { push(@OUT_LRL, $OUT_RL->[$iter]); }
          $iter++;
          if ($verbose > $COMMON::verbose2) { COMMON::show_progress($iter, $COMMON::progress1, $COMMON::progress2); }
       }
       if ($verbose > $COMMON::verbose2) { print STDERR ".$iter [DONE]\n"; }
       return(\@OUT_LRL, $Ftime, $Ctime);
    }
    elsif (($direction eq $COMMON::glrlmode) and ($mode == $SVMTAGGER::Mviterbi)) { #both LEFT-TO-RIGHT and RIGHT-TO-LEFT
       if ($M2 == 0) {
          ($OUT_LR, $GscoreLR, $FtimeLR, $CtimeLR) = do_tagging($M, $verbose, $COMMON::lrmode, $input_LR, $M1->{LR}, 0, $mode,$Unihan,$BS);# have modified  add $Unihan
          ($OUT_RL, $GscoreRL, $FtimeRL, $CtimeRL) = do_tagging($M, $verbose, $COMMON::rlmode, $input_LR, $M1->{RL}, 0, $mode,$Unihan,$BS);# have modified  add $Unihan
          $Ftime = $FtimeLR + $FtimeRL;
          $Ctime = $CtimeLR + $CtimeRL;
       }
       else {
          ($OUT_LR, $GscoreLR, $FtimeLR, $CtimeLR) = do_tagging($M, $verbose, $COMMON::lrmode, $input_LR, $M1->{LR}, $M2->{LR}, $mode,$Unihan,$BS);# have modified  add $Unihan
          ($OUT_RL, $GscoreRL, $FtimeRL, $CtimeRL) = do_tagging($M, $verbose, $COMMON::rlmode, $input_LR, $M1->{RL}, $M2->{RL}, $mode,$Unihan,$BS);# have modified  add $Unihan
          $Ftime = $FtimeLR + $FtimeRL;
          $Ctime = $CtimeLR + $CtimeRL;
       }
       if ($verbose > $COMMON::verbose2) {
          print STDERR "COMBINING LR/RL...\n";
          print STDERR "G-score(LR) = $GscoreLR\n";
          print STDERR "G-score(RL) = $GscoreRL\n";
       }
       if ($GscoreLR > $GscoreRL) { return($OUT_LR, $Ftime, $Ctime);  }
       else { return($OUT_RL, $Ftime, $Ctime);  }
    }
    else { die "[TAGGING] WRONG DIRECTION ($direction)\n"; }
}

sub do_tagging_LRL_2P
{
    #description _ responsible for performing tagging in a M0 strategy. (usual)
    #              right context is considered ambiguous.
    #param1  _ WHOLE MODELS
    #param2  _ VERBOSE 
    #param3  _ text direction "LR" for left-to-right ; "RL" for right-to-left; "LRL" for both
    #param4  _ input list reference
    #param5  _ 1st pass SVMT MODELS 1
    #param6  _ 1st pass SVMT MODELS 2
    #param7  _ 2nd pass SVMT MODELS 1
    #param8  _ 2nd pass SVMT MODELS 2
    #param9  _ mode: 0 -> do_normal :: 1 -> do_special :: 2 -> do_viterbi
    #param10  _ mode: 0 -> do_normal :: 1 -> do_special :: 2 -> do_viterbi
    #@return _ (output list reference, Ftime, Ctime)

    my $M = shift;
    my $verbose = shift;
    my $direction = shift;
    my $input_LR = shift;
    my $M1 = shift;
    my $M2 = shift;
    my $M3 = shift;
    my $M4 = shift;
    my $mode1 = shift;
    my $mode2 = shift;
    my $Unihan = shift;
    my $BS = shift;

    if ($verbose > $COMMON::verbose2) { print STDERR "1st PASS\n"; }
    my ($auxl, $Ftime, $Ctime) = do_tagging_LRL_1P($M, $verbose, $direction, $input_LR, $M1, $M2, $mode1,$Unihan,$BS);# have modified  add $Unihan
    if ($verbose > $COMMON::verbose2) { print STDERR "2nd PASS\n"; }
    return (do_tagging_LRL_1P($M, $verbose, $direction, $auxl, $M3, $M4, $mode2,$Unihan,$BS), $Ftime, $Ctime);# have modified  add $Unihan
}

# =============================================================================================
# =============================== EXPORTABLE METHODS ==========================================
# =============================================================================================

# -------------- EVALUATOR -----------------------------------------------------------------

sub SVMT_brief_eval
{
    #description _ briefly compute and print [known-acc amb-acc unk-acc overall-acc]
    #param1 _ model
    #param2 _ input filename
    #param3 _ output filename
    #param4 _ verbose (0/1)

    my $model = shift;
    my $input = shift;
    my $output = shift;
    my $verbose = shift;

    my $statistics = STATS::do_statistics($model, $input, $output, $verbose);
    STATS::print_results($statistics);
}

sub SVMT_deep_eval
{
    #description _ deeply evaluate and print SVMT results for several sets of words
    #param1 _ model
    #param2 _ input filename
    #param3 _ output filename
    #param4 _ mode
    #param5 _ verbose (0/1/2)
    #@return _ ($nhits, $nsamples)

    my $model = shift;
    my $input = shift;
    my $output = shift;
    my $mode = shift;
    my $verbose = shift;

    print "* ========================= SVMTeval report ==============================\n";
    print "* model               = [$model]\n";
    print "* testset (gold)      = [$input]\n";
    print "* testset (predicted) = [$output]\n";
    print "* ========================================================================\n";

    my $statistics = STATS::do_statistics($model, $input, $output, $verbose);
    STATS::print_stats_header($statistics);

    if ($mode == 0) { #complete report (everything)
       STATS::print_stats_ambU($statistics);
       STATS::print_stats_ambN($statistics);
       STATS::print_stats_ambK($statistics);
       STATS::print_stats_ambP($statistics);
    }
    elsif ($mode == 2) { #accuracy of known vs. unknown words
       STATS::print_stats_ambU($statistics);
    }
    elsif ($mode == 3) { #accuracy per level of ambiguity
       STATS::print_stats_ambN($statistics);
    }
    elsif ($mode == 4) { #accuracy per kind of ambiguity
       STATS::print_stats_ambK($statistics);
    } 
    elsif ($mode == 5) { #accuracy per part-of-speech 
       STATS::print_stats_ambP($statistics);
    }

    STATS::print_stats_overall($statistics);
}

# -------------- TAGGER --------------------------------------------------------------------

sub do_SVMT_file
{
   #description _ performs a given word per line tokenized corpus tagging.
   #              (includes known and unknown words)
   #              right-to-left direction
   #param1  _ MODEL NAME                 
   #param2  _ EPSILON threshold for W features relevance  [KNOWN WORDS] 
   #param3  _ OMEGA threshold for W features relevance  [UNKNOWN WORDS] 
   #param4  _ VERBOSE 
   #param5  _ text direction "LR" for left-to-right ; "RL" for right-to-left; "LRL" for both
   #param6  _ strategy --> 0: 1-pass ambiguous right
   #                       1: 2-passes (1)no-right (2)unambiguous-right
   #                       2: 1-pass with 2 models (1)no-right (2)unambiguous-right
   #                       3: 1-pass unsupervised learning
   #                       4: 1-pass unknown words seen in training of known words
   #                       5: 1-pass viterbi tagging (M0)
   #                       6: 1-pass viterbi tagging (M4)
   #param7  _ backup lexicon if available
   #param8  _ input filename
   #param9  _ output filename
   #param10 _ verbosity

   my $model = shift;
   my $epsilon = shift;
   my $omega = shift;
   my $verbose = shift;
   my $direction = shift;
   my $strategy = shift;
   my $blexicon = shift;
   my $input_file = shift;
   my $output_file = shift;
   my $verbose = shift;

   my $INPUT = new IO::File("< $input_file") or die "Couldn't open input file: $input_file\n";
   my $OUTPUT = new IO::File("> $output_file") or die "Couldn't open output file: $output_file\n";

   if ($verbose > $COMMON::verbose1) { print "---------------------------------------------------------------------------------------------\nTESTING MODEL ON <$input_file>...\n"; }

   # ======================== READING SVMT MODEL ===============================================
   my $M = SVMT_load($model, $strategy, $direction, $epsilon, $omega, $blexicon, $verbose);

   my $s = 0;
   my $stop = 0;
   while (!$stop) {
     my $input;
     my $in;
     ($input, $in, $stop) = ENTRY::read_sentence_file($INPUT);
     my ($out, $time) = SVMTAGGER::SVMT_tag($strategy, -1, 0, $COMMON::softmax1, $direction, $in, $M, 0);
     $s++;
     my $i = 0;
     my $iter = 0;
     while ($iter < scalar(@{$input})) {
	my @line = split($COMMON::in_valseparator, ${$input}[$iter]);
        if ((scalar(@line) == 0) or ($line[0] eq $COMMON::IGNORE)) {
	   print $OUTPUT $input->[$iter]."\n";
        }
        else {
           shift(@line);
           unshift(@line, $out->[$i]->get_pos);
           unshift(@line, $in->[$i]->get_word);
           print $OUTPUT join($COMMON::out_valseparator, @line)."\n";
           $i++;
        }
        $iter++;
      }
      if ($verbose > $COMMON::verbose2) { COMMON::show_progress($s, $COMMON::progress3, $COMMON::progress0); }
   }
   if ($verbose > $COMMON::verbose1) { print STDERR "...", $s - 1, " sentences [DONE]\n"; }
}

sub do_SVMT_list
{
   #description _ performs a given word per line tokenized corpus tagging.
   #              (includes known and unknown words)
   #              right-to-left direction
   #param1  _ MODEL NAME                 
   #param2  _ EPSILON threshold for W features relevance  [KNOWN WORDS] 
   #param3  _ OMEGA threshold for W features relevance  [UNKNOWN WORDS] 
   #param4  _ VERBOSE 
   #param5  _ text direction "LR" for left-to-right ; "RL" for right-to-left; "LRL" for both
   #param6  _ strategy --> 0: 1-pass ambiguous right
   #                       1: 2-passes (1)no-right (2)unambiguous-right
   #                       2: 1-pass with 2 models (1)no-right (2)unambiguous-right
   #                       3: 1-pass unsupervised learning
   #                       4: 1-pass unknown words seen in training of known words
   #                       5: 1-pass viterbi tagging (M0)
   #                       6: 1-pass viterbi tagging (M4)
   #param7  _ backup lexicon if available
   #param8  _ input list reference

   my $model = shift;
   my $epsilon = shift;
   my $omega = shift;
   my $verbose = shift;
   my $direction = shift;
   my $strategy = shift;
   my $blexicon = shift;
   my $INPUT_LR = shift;

   # ======================== READING SVMT MODEL ===============================================
   my $M = SVMT_load($model, $strategy, $direction, $epsilon, $omega, $blexicon, $verbose);

   # =========================== POS-TAGGING ===================================================
   return SVMT_tag($strategy, -1, 0, $COMMON::softmax1, $direction, $INPUT_LR, $M, $verbose);
}

sub SVMT_load
{
   #description _ responsible for loading SVMT (models + dictionary)
   #param1 _ model name
   #param2 _ strategy  --> 0: 1-pass ambiguous right
   #                       1: 2-passes (1)no-right (2)unambiguous-right
   #                       2: 1-pass with 2 models (0)ambiguous-right (2)unambiguous-right
   #                       3: 1-pass unsupervised learning
   #                       4: 1-pass unknown words seen in training of known words
   #                       5: 1-pass viterbi tagging (M0)
   #                       6: 1-pass viterbi tagging (M4)
   #param3 _ direction mode (LR/RL/LRL)
   #param4 _ EPSILON threshold for W features relevance  [KNOWN WORDS] 
   #param5 _ OMEGA threshold for W features relevance  [UNKNOWN WORDS] 
   #param6 _ backup lexicon if available
   #param7 _ VERBOSE 
   #@return _ SVMT models

   my $model = shift;
   my $strategy = shift;
   my $direction = shift;
   my $epsilon = shift;
   my $omega = shift;
   my $blexicon = shift;
   my $verbose = shift;

   my %M;  #SVMT COMPLETE MODEL

   # ========================== READING DICTIONARY =============================================
   my $dict;
   if ($blexicon ne "") { #joining backup lexicon
      srand();
      my $xpnddict = $COMMON::DICTEXT.".".rand(100000);
      if ($verbose) { print STDERR "EXPANDING DICTIONARY <".$model.".".$COMMON::DICTEXT."> WITH <".$blexicon."> ONTO <".$xpnddict.">\n"; }
      expand_dictionary($model.".".$COMMON::DICTEXT, $blexicon, $xpnddict, $verbose); 
      $dict = SVMT_load_dict($xpnddict, $model.".".$COMMON::AMBPEXT, $model.".".$COMMON::UNKPEXT, $verbose);
      system "rm -f $xpnddict";
   }
   else {
      $dict = SVMT_load_dict($model.".".$COMMON::DICTEXT, $model.".".$COMMON::AMBPEXT, $model.".".$COMMON::UNKPEXT, $verbose);
   }
   $M{dict} = $dict;

   # ========================== READING SLIDING WINDOW INFO ====================================
   $M{WS} = COMMON::read_list($model.".".$COMMON::WINEXT);
   # ========================== READING POS lists ==============================================
   my $rambp = COMMON::read_list($model.".".$COMMON::AMBPEXT);
   my $runkp = COMMON::read_list($model.".".$COMMON::UNKPEXT);
   $M{rAP} = COMMON::do_hash($rambp, "");
   $M{rUP} = COMMON::do_hash($runkp, "");

   # ========================== READING MODELS and feature sets ================================
   if ($strategy == $COMMON::st0) {
      #one-pass [right context is ambiguous] (default)
      $M{A0k} = SVMTAGGER::read_fs($model.".".$COMMON::A0EXT, $M{WS});
      $M{A0u} = SVMTAGGER::read_fs($model.".".$COMMON::A0EXT.".".$COMMON::unkext, $M{WS});
      $M{P1} = SVMT_load_models($model, $direction, $COMMON::mode0, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A0k}, $M{A0u});
   }
   elsif ($strategy == $COMMON::st1) {
      #two-passes [1st no-right :: 2nd unambiguous-right]
      $M{A2k} = SVMTAGGER::read_fs($model.".".$COMMON::A2EXT, $M{WS});
      $M{A2u} = SVMTAGGER::read_fs($model.".".$COMMON::A2EXT.".".$COMMON::unkext, $M{WS});
      $M{A1k} = SVMTAGGER::read_fs($model.".".$COMMON::A1EXT, $M{WS});
      $M{A1u} = SVMTAGGER::read_fs($model.".".$COMMON::A1EXT.".".$COMMON::unkext, $M{WS});
      $M{P1} = SVMT_load_models($model, $direction, $COMMON::mode2, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A2k}, $M{A2u});
      $M{P2} = SVMT_load_models($model, $direction, $COMMON::mode1, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A1k}, $M{A1u});
   }
   elsif ($strategy == $COMMON::st2) {
      #one-pass [unknown words are special]
      $M{A0k} = SVMTAGGER::read_fs($model.".".$COMMON::A0EXT, $M{WS});
      $M{A0u} = SVMTAGGER::read_fs($model.".".$COMMON::A0EXT.".".$COMMON::unkext, $M{WS});
      $M{A2k} = SVMTAGGER::read_fs($model.".".$COMMON::A2EXT, $M{WS});
      $M{A2u} = SVMTAGGER::read_fs($model.".".$COMMON::A2EXT.".".$COMMON::unkext, $M{WS});
      $M{P1} = SVMT_load_models($model, $direction, $COMMON::mode0, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A0k}, $M{A0u});
      $M{P2} = SVMT_load_models($model, $direction, $COMMON::mode2, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A2k}, $M{A2u});
   }
   elsif ($strategy == $COMMON::st3) {
      #one-pass unsupervised learning models
      $M{A3k} = SVMTAGGER::read_fs($model.".".$COMMON::A3EXT, $M{WS});
      $M{A3u} = SVMTAGGER::read_fs($model.".".$COMMON::A3EXT.".".$COMMON::unkext, $M{WS});
      $M{P1} = SVMT_load_models($model, $direction, $COMMON::mode3, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A3k}, $M{A3u});
   }
   elsif ($strategy == $COMMON::st4) {
      #one-pass [right context is ambiguous] [there are unknown words on training]
      $M{A4k} = SVMTAGGER::read_fs($model.".".$COMMON::A4EXT, $M{WS});
      $M{A4u} = SVMTAGGER::read_fs($model.".".$COMMON::A4EXT.".".$COMMON::unkext, $M{WS});
      $M{P1} = SVMT_load_models($model, $direction, $COMMON::mode4, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A4k}, $M{A4u});
   }
   elsif ($strategy == $COMMON::st5) {
      #one-pass TRELLIS [unambiguous-right] VITERBI sentence-level M0
      $M{A0k} = SVMTAGGER::read_fs($model.".".$COMMON::A0EXT, $M{WS});
      $M{A0u} = SVMTAGGER::read_fs($model.".".$COMMON::A0EXT.".".$COMMON::unkext, $M{WS});
      $M{P1} = SVMT_load_models($model, $direction, $COMMON::mode0, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A0k}, $M{A0u});
   }
   elsif ($strategy == $COMMON::st6) {
      #one-pass TRELLIS [unambiguous-right] VITERBI sentence-level M4
      #$M{A1k} = SVMTAGGER::read_fs($model.".".$COMMON::A1EXT, $M{WS});
      #$M{A1u} = SVMTAGGER::read_fs($model.".".$COMMON::A1EXT.".".$COMMON::unkext, $M{WS});
      #$M{P1} = SVMT_load_models($model, $direction, $COMMON::mode1, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A1k}, $M{A1u});
      $M{A4k} = SVMTAGGER::read_fs($model.".".$COMMON::A4EXT, $M{WS});
      $M{A4u} = SVMTAGGER::read_fs($model.".".$COMMON::A4EXT.".".$COMMON::unkext, $M{WS});
      $M{P1} = SVMT_load_models($model, $direction, $COMMON::mode4, $epsilon, $omega, $verbose, $rambp, $runkp, $M{A4k}, $M{A4u});
   }

   return \%M;
}

sub SVMT_tag
{
   #description _ given a list of tokens, performs the SVM-based pos-tagging.
   #param1 _ strategy  --> 0: 1-pass ambiguous right
   #                       1: 2-passes (1)no-right (2)unambiguous-right
   #                       2: 1-pass with 2 models (1)no-right (2)unambiguous-right
   #                       3: 1-pass unsupervised learning
   #                       4: 1-pass unknown words seen in training of known words
   #                       5: 1-pass viterbi tagging (M0)
   #                       6: 1-pass viterbi tagging (M4)
   #param2 _ beam count cutoff [only applicable under sentence-level strategy] 
   #param3 _ beam search ratio [only applicable under sentence-level strategy] 
   #param4 _ softmax mode 0: do nothing
   #       _              1: do ln(e^score(i) / [sum:1<=j<=N:[e^score(j)]])
   #param5 _ direction mode (LR/RL/LRL)
   #param6 _ input list reference
   #param7 _ SVMT models
   #param8 _ VERBOSE
   #@return _ (output list reference, Ftime, Ctime)

   my $strategy = shift;
   my $nbeams = shift;
   my $bratio = shift;
   my $softmax = shift;
   my $direction = shift;
   my $INPUT_LR = shift;
   my $M = shift;
   my $verbose = shift;

   my $out;
   my $Ftime;
   my $Ctime;
   
   my $Unihan = read_unihan();
   my $BS = read_bs();

   if (!(defined($M))) { die "[SVMT] model is unavailable!!\n"; }

   $M->{nbeams} = $nbeams;
   $M->{softmax} = $softmax;
   $M->{bratio} = $bratio;

   if (scalar(@{$INPUT_LR}) > 0) { # list isn't empty
      if (($strategy == $COMMON::st0) or ($strategy == $COMMON::st3) or ($strategy == $COMMON::st4)) {
         #one-pass default
         ($out, $Ftime, $Ctime) = do_tagging_LRL_1P($M, $verbose, $direction, $INPUT_LR, $M->{P1}, 0, $SVMTAGGER::Mdefault,$Unihan,$BS);# have modified  add $Unihan
      }
      elsif ($strategy == $COMMON::st1) { #two-passes [1st no-right :: 2nd unambiguous-right]
         ($out, $Ftime, $Ctime) = do_tagging_LRL_2P($M, $verbose, $direction, $INPUT_LR, $M->{P1}, 0, $M->{P2}, 0, $SVMTAGGER::Mdefault, $SVMTAGGER::Mdefault,$Unihan,$BS);# have modified  add $Unihan
      }
      elsif ($strategy == $COMMON::st2) { #one-pass [unknown words are special]
         ($out, $Ftime, $Ctime) = do_tagging_LRL_1P($M, $verbose, $direction, $INPUT_LR, $M->{P1}, $M->{P2}, $SVMTAGGER::Mspecial,$Unihan,$BS);# have modified  add $Unihan
      }
      elsif (($strategy == $COMMON::st5) or ($strategy == $COMMON::st6)) {
         #one-pass sentence-level likelihood VITERBI [ambiguous right]
         ($out, $Ftime, $Ctime) = do_tagging_LRL_1P($M, $verbose, $direction, $INPUT_LR, $M->{P1}, 0, $SVMTAGGER::Mviterbi,$Unihan,$BS);# have modified  add $Unihan
      }
   }

   my @time = ($Ftime, $Ctime);

   return ($out, \@time);
}

sub SVMT_prepare_input
{
    #description _ transforms a list of words into a list of entries
    #param1  _ list of tokens
    #@return _ list of entries

    my $tokens = shift;

    my @entries;
    my $i = 0;
    while ($i < scalar(@{$tokens})) {
      if ($tokens->[$i] ne "") {
         my @l = split(" ", $tokens->[$i]);
         my $word = shift(@l);
         my $elem = new ENTRY($word, $COMMON::emptypos, 0, \@l);
         push(@entries, $elem);
      }
      $i++;
    }

    return \@entries;
}

# -------------- LEARNER -------------------------------------------------------------------

sub learn_known
{
    #description _ responsible for learning an SVMT model for known tokens.
    #param1  _ model name
    #param2  _ training set filename
    #param3  _ sample set filename
    #param4  _ sample set binary mapping filename
    #param5  _ dictionary object reference
    #param6  _ model type
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unkown words on training)    
    #param7  _ model type file EXTENSION
    #param8  _ model direction
    #param9  _ model direction file EXTENSION
    #param10 _ configuration hash reference
    #param11 _ remake? 1 -> features are extracted (out from scratch) from trainset
    #                  0 -> existing sample file is used (also mapping)
    #param12 _ report file
    #param13 _ verbosity [0..3]

    my $model = shift;
    my $trainset = shift;
    my $smplset = shift;
    my $smplmap = shift;
    my $rdict = shift;
    my $mode = shift;
    my $modext = shift;
    my $direction = shift;
    my $dirext = shift;
    my $config = shift;
    my $remakeF = shift;
    my $report = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $smpldsf = $smplset.".".$COMMON::DSFEXT;

    # ----------------- KNOWN WORDS ----------------------------------------------------------
    if ($verbose > $COMMON::verbose1) { COMMON::report($report, "---------------- KNOWN WORDS... [MODE = $mode :: DIRECTON = $direction] -------------------\n"); }

    if ($remakeF) {
       if ($mode == $COMMON::mode4) {
          SVMTAGGER::do_attribs_kn_unk($model, $trainset, $smplset, $direction, $mode, $config, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
       else {
          SVMTAGGER::do_attribs_kn($trainset, $rdict, $smplset, $direction, $mode, $config, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
       my $rM = MAPPING::make_mapping($smplset, $report, $verbose);
       if ($config->{F} == 1) {
          $rM = MAPPING::filter_mapping($rM, $config->{fmin}, $config->{maxmapsize}, $report, $verbose);
       }
       MAPPING::write_mapping($rM, $smplmap, $report, $verbose);
       MAPPING::map_set($smplset, $smplmap, $smpldsf, $report, $verbose);
    }

    SVMTAGGER::do_learn($rdict, $smplmap, $config->{AP}, $model, $modext, $dirext, $smpldsf, $config->{Ck}, 0, 0, $mode, $config->{SVMDIR}, $config->{Eratio}, $report, $verbose);

    if ($config->{rmfiles}) { system "rm -f $smplset"; system "rm -f $smpldsf";}
}

sub learn_unknown
{
    #description _ responsible for learning an SVMT model for unknown tokens.
    #param1  _ model name
    #param2  _ training set filename
    #param3  _ sample set filename
    #param4  _ sample set binary mapping filename
    #param5  _ dictionary object reference
    #param6  _ model type
    #param7  _ model type file EXTENSION
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unkown words on training)    
    #param8  _ model direction
    #param9  _ model direction file EXTENSION
    #param10 _ configuration hash reference
    #param11 _ remake? 1 -> features are extracted (out from scratch) from trainset
    #                  0 -> existing sample file is used (also mapping)
    #param12 _ report file
    #param13 _ verbosity [0..3]

    my $model = shift;
    my $trainset = shift;
    my $unksmplset = shift;
    my $unksmplmap = shift;
    my $rdict = shift;
    my $mode = shift;
    my $modext = shift;
    my $direction = shift;
    my $dirext = shift;
    my $config = shift;
    my $remakeF = shift;
    my $report = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $unksmpldsf = $unksmplset.".".$COMMON::DSFEXT;

    # --------------- UNKNOWN WORDS ----------------------------------------------------------
    if ($verbose > $COMMON::verbose1) { COMMON::report($report, "---------------- UNKNOWN WORDS... [MODE = $mode :: DIRECTON = $direction] -------------------\n"); }

    if ($remakeF) {
       SVMTAGGER::do_attribs_unk($model, $trainset, $rdict, $unksmplset, $direction, $mode, $config, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       my $rUM = MAPPING::make_mapping($unksmplset, $report, $verbose);
       if ($config->{F} == 1) {
          $rUM = MAPPING::filter_mapping($rUM, $config->{fmin}, $config->{maxmapsize}, $report, $verbose);
       }
       MAPPING::write_mapping($rUM, $unksmplmap, $report, $verbose);
       MAPPING::map_set($unksmplset, $unksmplmap, $unksmpldsf, $report, $verbose);
    }

    SVMTAGGER::do_learn_unk($unksmplmap, $config->{UP}, $model, $modext, $dirext, $unksmpldsf, $config->{Cu}, 0, 0, $mode, $config->{SVMDIR}, $report, $verbose);
    if ($config->{rmfiles}) { system "rm -f $unksmplset"; system "rm -f $unksmpldsf"; }
}

sub merge_models
{
    #description _ responsible for merging SVMT models.
    #param1  _ model name
    #param2  _ model type file EXTENSION
    #param3  _ model direction file EXTENSION
    #param4  _ sample set binary mapping filename for known words
    #param5  _ sample set binary mapping filename for unknown words
    #param6  _ configuration hash reference
    #param7  _ merged file extension
    #param8  _ which models to merge (0 -> only known words : 1 -> only unknown words : 2 -> both
    #param9  _ report file
    #param10 _ verbosity [0..3]

    my $model = shift;
    my $modext = shift;
    my $dirext = shift;
    my $smplmap = shift;
    my $unksmplmap = shift;
    my $config = shift;
    my $MRGEXT = shift;
    my $which = shift;
    my $report = shift;
    my $verbose = shift;

    # -------------------------------------------------------------------------------------------
    # ----------------------- WRITING MERGED MODELS ---------------------------------------------

    if (($which == 0) or ($which == 2)) {
       # -------------------------- MAPPING -------------------------------------------------------- 
       if ($verbose > $COMMON::verbose1) { print "READING MAPPING FOR KNOWN WORDS...<$smplmap>\n"; }
       my $rmap = MAPPING::read_mapping($smplmap);
       # -------------------------- SVM ------------------------------------------------------------
       # models are read [and optionally irrelevant weights may be filtered out]
       if ($verbose > $COMMON::verbose1) { print "READING SVM-MODELS FOR KNOWN WORDS...\n"; }
       my ($rW, $rB) = SVM::read_models($model, $config->{AP}, "$modext.$dirext", $config->{Kfilter}, $rmap, $verbose);
       # -------------------------- MODEL/MAPPING merging optimization -----------------------------
       if ($verbose > $COMMON::verbose1) { print "MERGING MAPPING and MODELS FOR KNOWN WORDS...\n"; }
       my $KNMODEL = SVM::merge_models($rmap, $rW);
       if ($verbose > $COMMON::verbose1) { print "WRITING MERGED MODELS FOR KNOWN WORDS...<$model.$modext.$dirext.$MRGEXT>\n"; }
       SVM::write_merged_models($KNMODEL, $rB, $config, "$model.$modext.$dirext.$MRGEXT", 0);
       if ($config->{rmfiles}) {
          system "rm -f $smplmap";
          system "rm -f $model.$modext.$dirext.$COMMON::Bext";
          system "rm -f $model.$modext.$dirext.$COMMON::Wext";
       }
    }
    if (($which == 1) or ($which == 2)) {
       # -------------------------- MAPPING -------------------------------------------------------- 
       if ($verbose > $COMMON::verbose1) { print "READING MAPPING FOR UNKNOWN WORDS...<$unksmplmap>\n"; }
       my $runkmap = MAPPING::read_mapping($unksmplmap); 
       if ($verbose > $COMMON::verbose1) { print "READING SVM-MODELS FOR UNKNOWN WORDS...\n"; }
       # -------------------------- SVM ------------------------------------------------------------
       # models are read [and optionally irrelevant weights may be filtered out]
       my ($runkW, $runkB) = SVM::read_models($model, $config->{UP}, "$COMMON::unkext.$modext.$dirext", $config->{Ufilter}, $runkmap, $verbose);
       # -------------------------- MODEL/MAPPING merging optimization -----------------------------
       if ($verbose > $COMMON::verbose1) { print "MERGING MAPPING and MODELS FOR UNKNOWN WORDS...\n"; }
       my $UNKMODEL = SVM::merge_models($runkmap, $runkW);
       if ($verbose > $COMMON::verbose1) { print "WRITING MERGED MODELS FOR UNKNOWN WORDS...<$model.$COMMON::unkext.$modext.$dirext.$MRGEXT>\n"; }
       SVM::write_merged_models($UNKMODEL, $runkB, $config, "$model.$COMMON::unkext.$modext.$dirext.$MRGEXT", 1);
       if ($config->{rmfiles}) {
          system "rm -f $unksmplmap";
          system "rm -f $model.$COMMON::unkext.$modext.$dirext.$COMMON::Wext";
          system "rm -f $model.$COMMON::unkext.$modext.$dirext.$COMMON::Bext";
       }
    }
}

sub build_SVMT
{
    #description _ responsible for building SVMT models [given a current setting].
    #param1  _ model name
    #param2  _ training set filename
    #param3  _ dictionary object reference
    #param4  _ model type
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unkown words on training)    
    #param5  _ model direction
    #param6  _ configuration hash reference
    #param7  _ which models to build (0 -> only known words : 1 -> only unknown words : 2 -> both
    #param8  _ remake? 1 -> features are extracted (out from scratch) from trainset
    #          KNOWN   0 -> existing sample file is used (also mapping)
    #param9  _ remake? 1 -> features are extracted (out from scratch) from trainset
    #          UNKNOWN 0 -> existing sample file is used (also mapping)
    #param10 _ report file
    #param11 _ verbosity [0..3]


    my $model = shift;
    my $trainset = shift;
    my $rdict = shift;
    my $mode = shift;
    my $direction = shift;
    my $config = shift;
    my $which = shift;
    my $remakeFK = shift;
    my $remakeFU = shift;
    my $report = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $dirext;
    if ($direction eq $COMMON::lrmode) { $dirext .= "$COMMON::lrext"; }
    elsif ($direction eq $COMMON::rlmode) { $dirext .= "$COMMON::rlext"; }
    my $modext = SVMTAGGER::find_mext($mode);

    if ($remakeFK) {
       if (-e "$model.$modext.$dirext*") {
          if ($verbose > $COMMON::verbose1) {
             print "ERASING previous work on same models KNOWN < $model :: $modext :: $dirext >\n";
          }
          system "rm -f $model.$modext.$dirext*";
       }
    }
    if ($remakeFU > $COMMON::verbose1) {
       if (-e "$model.$COMMON::unkext.$modext.$dirext.*") {
          if ($verbose) {
             print "ERASING previous work on same models UNKNOWN < $model :: $COMMON::unkext :: $modext :: $dirext >\n";
          }
          system "rm -f $model.$COMMON::unkext.$modext.$dirext.*";
       }
    }

    my $smplset = $model.".".$modext.".".$dirext.".".$COMMON::smplext;  #SAMPLE SET
    my $smplmap = $smplset.".".$COMMON::mapext;                         #SAMPLE MAPPING
    my $unksmplset = $model.".".$COMMON::unkext.".".$modext.".".$dirext.".".$COMMON::smplext;  #SAMPLE SET
    my $unksmplmap = $unksmplset.".".$COMMON::mapext;                                             #SAMPLE MAPPING

    if (($which == 0) or ($which == 2)) {
       SVMTAGGER::learn_known($model, $trainset, $smplset, $smplmap, $rdict, $mode, $modext, $direction, $dirext, $config, $remakeFK, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
    }
    if (($which == 1) or ($which == 2)) {
       SVMTAGGER::learn_unknown($model, $trainset, $unksmplset, $unksmplmap, $rdict, $mode, $modext, $direction, $dirext, $config, $remakeFU, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
    }
    SVMTAGGER::merge_models($model, $modext, $dirext, $smplmap, $unksmplmap, $config, $COMMON::MRGEXT, $which, $report, $verbose);
}

sub adjust_C
{
    #description _ responsible for tuning C parameter for SVMT models.
    #param1  _ model name
    #param2  _ training set filename
    #param3  _ dictionary object reference
    #param4  _ model type
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unkown words on training)    
    #param5  _ model direction
    #param6  _ configuration hash reference
    #param7  _ C parameter tuning options
    #          C:begin:end:n_iters:n_segments:[log|nolog]:[V|CV]:CV_n_folders
    #          e.g. C:0.001:10:3:5:log:V
    #          e.g. C:0.001:10:3:5:nolog:V
    #          e.g. C:0.001:10:3:5:log:CV:10
    #param8  _ validation set
    #param9  _ adjust C for (0) known words (1) unknown words
    #param10 _ report file
    #param11 _ verbosity [0..3]
    #@return _ (optimal C, remake features?)

    my $model = shift;
    my $trainset = shift;
    my $rdict = shift;
    my $mode = shift;
    my $direction = shift;
    my $config = shift;
    my $CTopt = shift;
    my $valset = shift;
    my $unknown = shift;
    my $report = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my @CToptions = split(/;/, $CTopt);
    my $MIN_C = $CToptions[1];
    my $MAX_C = $CToptions[2];
    my $MAX_DEPTH = $CToptions[3];
    my $nSEGMENTS = $CToptions[4];
    my $LOG = ($CToptions[5] eq "log");

    if ($verbose) {
       COMMON::report($report, "*********************************************************************************************\nC-PARAMETER TUNING\non <$trainset>\non <MODE $mode> <DIRECTION $direction> [".($unknown? "UNKNOWN" : "KNOWN")."]\nC-RANGE = [$MIN_C..$MAX_C] :: [$CToptions[5]] :: #LEVELS = ".$MAX_DEPTH." :: SEGMENTATION RATIO = ".$nSEGMENTS."\n*********************************************************************************************\n");
    }
    my $dirext;
    my $strategy = SVMTAGGER::find_strategy($mode);
    if ($direction eq $COMMON::lrmode) { $dirext .= "$COMMON::lrext"; }
    elsif ($direction eq $COMMON::rlmode) { $dirext .= "$COMMON::rlext"; }

    my $modext = SVMTAGGER::find_mext($mode);

    #BUILIDING COOPERATIVE MODELS [working together] ONLY IF NOT EXISTED
    if ($mode == $COMMON::mode1) {
       if ($verbose) { COMMON::report($report, "[INIT] BUILDING COOPERATIVE MODELS [M1 -> M2]...\n"); }
       if ((!(-e "$model.$COMMON::M2EXT.$dirext.$COMMON::MRGEXT")) or (!(-e "$model.$COMMON::unkext.$COMMON::M2EXT.$dirext.$COMMON::MRGEXT"))) { #build cooperative models [working together] only if they didn't existed
          SVMTAGGER::build_SVMT($model, $trainset, $rdict, $COMMON::mode2, $direction, $config, 2, 1, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
    }
    elsif ($mode == $COMMON::mode2) {
       if ($verbose) { COMMON::report($report, "[INIT] BUILDING COOPERATIVE MODELS [M2 -> M1]...\n"); }
       if ((!(-e "$model.$COMMON::M1EXT.$dirext.$COMMON::MRGEXT")) or (!(-e "$model.$COMMON::unkext.$COMMON::M1EXT.$dirext.$COMMON::MRGEXT"))) {
          SVMTAGGER::build_SVMT($model, $trainset, $rdict, $COMMON::mode1, $direction, $config, 2, 1, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
    }

    my $smplset = $model.".".$modext.".".$dirext.".".$COMMON::smplext;  #SAMPLE SET
    my $smplmap = $smplset.".".$COMMON::mapext;                         #SAMPLE MAPPING
    my $unksmplset = $model.".".$COMMON::unkext.".".$modext.".".$dirext.".".$COMMON::smplext;  #SAMPLE SET
    my $unksmplmap = $unksmplset.".".$COMMON::mapext;                                             #SAMPLE MAPPING

    my $accuracy = 0;      #final accuracy
    my $maxC = $MIN_C;     #C found
    my $maxiter = 0;       #where C was found
    my $maxdepth = 0;      #wher C was found
    my $factor = ($MAX_C - $MIN_C) / ($nSEGMENTS);   #factor to scale greedy exploration

    if ($LOG) { $factor = 10; } #first step goes logarithmically

    my $maxacc = -1;
    my $depth = 0;
    my $C;

    my $remakeF = 1; #REMAKE FEATURES? [only the first iteration]
    while ($depth < $MAX_DEPTH){
       if ($verbose) {
          COMMON::report($report, "\n=============================================================================================\nLEVEL = $depth :: C-RANGE = [$MIN_C..$MAX_C] :: FACTOR = [".($LOG? "*" : "+")." $factor ]\n=============================================================================================\n"); } 

       my $i = 0;
       $C = $MIN_C;

       while (((sprintf("%.4f", $MAX_C - $C) + 0) >= 0) && ($C >= 0)) {

          if (($depth == 0) and ($i == 0)) { #BUILDING COMPLEMENTARY MODELS - only if they didn't exist
             if ($verbose) { COMMON::report($report, "\n[INIT] BUILDING COMPLEMENTARY MODELS [known <-> unknown]...\n"); }
             if ($unknown) { #train complementary model (one not being tuned) outside the loop
                if (!(-e "$model.$modext.$dirext.$COMMON::MRGEXT")) {
                   SVMTAGGER::learn_known($model, $trainset, $smplset, $smplmap, $rdict, $mode, $modext, $direction, $dirext, $config, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
                   SVMTAGGER::merge_models($model, $modext, $dirext, $smplmap, $unksmplmap, $config, $COMMON::MRGEXT, !$unknown, $report, $verbose);
   	        }
             }
             else {
                if (!(-e "$model.$COMMON::unkext.$modext.$dirext.$COMMON::MRGEXT")) {
                   SVMTAGGER::learn_unknown($model, $trainset, $unksmplset, $unksmplmap, $rdict, $mode, $modext, $direction, $dirext, $config, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
                   SVMTAGGER::merge_models($model, $modext, $dirext, $smplmap, $unksmplmap, $config, $COMMON::MRGEXT, !$unknown, $report, $verbose);
                }
             }
	  }

          $i++;
          if ($verbose) { COMMON::report($report, "\n---------------------------------------------------------------------------------------------\n******************************** level - $depth : ITERATION ".($i - 1)." - C = $C - [M$mode :: $direction]\n---------------------------------------------------------------------------------------------\n\n"); }

          if ($unknown) {
             $config->{Cu} = $C;
             $remakeF = (!(-e "$model.$COMMON::unkext.$modext.$dirext.$COMMON::smplext.$COMMON::DSFEXT"));
             SVMTAGGER::learn_unknown($model, $trainset, $unksmplset, $unksmplmap, $rdict, $mode, $modext, $direction, $dirext, $config, $remakeF, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
 	  }
          else {
             $config->{Ck} = $C;
             $remakeF = (!(-e "$model.$modext.$dirext.$COMMON::smplext.$COMMON::DSFEXT"));
             SVMTAGGER::learn_known($model, $trainset, $smplset, $smplmap, $rdict, $mode, $modext, $direction, $dirext, $config, $remakeF, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
	  }
          if ($direction eq $COMMON::rlmode) { system "rm -f $model.$COMMON::revext"; }
          if ($remakeF == 1) { $remakeF = 0; }

          #backup previous models (if they existed)
          if (-e "$model.$modext.$dirext.$COMMON::MRGEXT") { system "cp $model.$modext.$dirext.$COMMON::MRGEXT $model.$modext.$dirext.$COMMON::mrgext" == 0 or die "system failed: $?";
	  } 
          if (-e "$model.$COMMON::unkext.$modext.$dirext.$COMMON::MRGEXT") { system "cp $model.$COMMON::unkext.$modext.$dirext.$COMMON::MRGEXT $model.$COMMON::unkext.$modext.$dirext.$COMMON::mrgext" == 0 or die "system failed: $?"; }
          SVMTAGGER::merge_models($model, $modext, $dirext, $smplmap, $unksmplmap, $config, $COMMON::MRGEXT, $unknown, $report, $verbose);
          srand();
          my $output_file = "$model.$modext.$dirext.$COMMON::clsfext.".rand(100000);
          SVMTAGGER::do_SVMT_file($model, 0, 0, 1, $direction, $strategy, "", $valset, $output_file, $verbose);
          my $stats = STATS::do_statistics($model, $valset, $output_file, $verbose);
          system "rm -f $output_file";
          my ($acckn, $accamb, $accunk, $accuracy) = STATS::get_accuracies($stats);

          if ($accuracy > $maxacc) { $maxacc = $accuracy; $maxiter = $i; $maxdepth = $depth; $maxC = $C; }
	  else { #restore previous models
	     if (-e "$model.$modext.$dirext.$COMMON::mrgext") { system "mv $model.$modext.$dirext.$COMMON::mrgext $model.$modext.$dirext.$COMMON::MRGEXT" == 0 or die "system failed: $?"; }
             if (-e "$model.$COMMON::unkext.$modext.$dirext.$COMMON::mrgext") { system "mv $model.$COMMON::unkext.$modext.$dirext.$COMMON::mrgext $model.$COMMON::unkext.$modext.$dirext.$COMMON::MRGEXT" == 0 or die "system failed: $?"; }
	  }

          if ($verbose) {
	     COMMON::report($report, "OVERALL ACCURACY [Ck = $config->{Ck} :: Cu = $config->{Cu}] : $accuracy%\nKNOWN [ $acckn% ] AMBIG.KNOWN [ $accamb% ] UNKNOWN [ $accunk% ]\nMAX ACCURACY -> $maxacc :: C-value = $maxC :: depth = $maxdepth :: iter = $maxiter\n");
	  }
          #update on C
          if (($depth == 0) && ($LOG == 1)) { $C = $C * $factor; }
          else { $C = $C + $factor; }
       }

       #update on EXPLORATION INTERVAL
       if (($depth == 0) && ($LOG)) { $MIN_C = $maxC / 2; $MAX_C = $maxC * $factor / 2; }
       else { $MIN_C = $maxC - ($factor / 2); $MAX_C = $maxC + ($factor / 2); }

       #update on EXPLORATION FACTOR
       my $olddepth = $depth;
       $depth++;
       if ($depth < $MAX_DEPTH) {
          $factor = ($MAX_C - $MIN_C) / $nSEGMENTS; 
          if ($verbose) {
 	     COMMON::report($report, "\nJUMPING FROM LEVEL $olddepth INTO LEVEL $depth : C = $maxC : built up on iter = $maxiter\n"); } 
       }
    }

    return ($maxC, $remakeF);
}

sub adjust_C_CV
{
    #description _ responsible for tuning C parameter for SVMT models.
    #param1  _ model name
    #param2  _ training set filename
    #param3  _ dictionary object reference
    #param4  _ number of folders
    #param5  _ model type
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unkown words on training)    
    #param6  _ model direction
    #param7  _ configuration hash reference
    #param8  _ C parameter tuning options
    #          C:begin:end:n_iters:n_segments:[log|nolog]:[V|CV]:CV_n_folders
    #          e.g. C:0.001:10:3:5:log:V
    #          e.g. C:0.001:10:3:5:nolog:V
    #          e.g. C:0.001:10:3:5:log:CV:10
    #param9  _ adjust C for (0) known words (1) unknown words
    #param10 _ remake CV folders?
    #param11 _ report file
    #param12 _ verbosity [0..3]
    #@return _ optimal C

    my $model = shift;
    my $trainset = shift;
    my $rdict = shift;
    my $nfolders = shift;
    my $mode = shift;
    my $direction = shift;
    my $config = shift;
    my $CTopt = shift;
    my $unknown = shift;
    my $remakeFLDS = shift;
    my $report = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    my @CToptions = split(/;/, $CTopt);
    my $MIN_C = $CToptions[1];
    my $MAX_C = $CToptions[2];
    my $MAX_DEPTH = $CToptions[3];
    my $nSEGMENTS = $CToptions[4];
    my $LOG = ($CToptions[5] eq "log");

    if ($verbose) {
       COMMON::report($report, "*********************************************************************************************\nC-PARAMETER TUNING by $nfolders-fold CROSS-VALIDATION\non <$trainset>\non <MODE $mode> <DIRECTION $direction> [".($unknown? "UNKNOWN" : "KNOWN")."]\nC-RANGE = [$MIN_C..$MAX_C] :: [$CToptions[5]] :: #LEVELS = ".$MAX_DEPTH." :: SEGMENTATION RATIO = ".$nSEGMENTS."\n*********************************************************************************************\n");
    }
    my $dirext;
    my $strategy = SVMTAGGER::find_strategy($mode);
    if ($direction eq $COMMON::lrmode) { $dirext .= "$COMMON::lrext"; }
    elsif ($direction eq $COMMON::rlmode) { $dirext .= "$COMMON::rlext"; }

    my $modext = SVMTAGGER::find_mext($mode);
    if ($mode == $COMMON::mode1) {
       if ($verbose) { COMMON::report($report, "[INIT] BUILDING COOPERATIVE MODELS [M1 -> M2]...\n"); }
       if ((!(-e "$model.$COMMON::M2EXT.$dirext.$COMMON::MRGEXT")) or (!(-e "$model.$COMMON::unkext.$COMMON::M2EXT.$dirext.$COMMON::MRGEXT"))) { #build cooperative models [working together] only if they didn't existed
          SVMTAGGER::build_SVMT($model, $trainset, $rdict, $COMMON::mode2, $direction, $config, 2, 1, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
    }
    elsif ($mode == $COMMON::mode2) {
       if ($verbose) { COMMON::report($report, "[INIT] BUILDING COOPERATIVE MODELS [M2 -> M1]...\n"); }
       if ((!(-e "$model.$COMMON::M1EXT.$dirext.$COMMON::MRGEXT")) or (!(-e "$model.$COMMON::unkext.$COMMON::M1EXT.$dirext.$COMMON::MRGEXT"))) { #build cooperative models [working together] only if not existed
          SVMTAGGER::build_SVMT($model, $trainset, $rdict, $COMMON::mode1, $direction, $config, 2, 1, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
    }

    my $smplset = $model.".".$modext.".".$dirext.".".$COMMON::smplext;  #SAMPLE SET
    my $smplmap = $smplset.".".$COMMON::mapext;                         #SAMPLE MAPPING
    my $unksmplset = $model.".".$COMMON::unkext.".".$modext.".".$dirext.".".$COMMON::smplext; #SAMPLE SET
    my $unksmplmap = $unksmplset.".".$COMMON::mapext;                                         #SAMPLE MAPPING

    my $accuracy = 0;      #final accuracy
    my $maxC = $MIN_C;     #C found
    my $maxiter = 0;       #where C was found
    my $maxdepth = 0;      #wher C was found
    my $factor = ($MAX_C - $MIN_C) / ($nSEGMENTS);   #factor to scale greedy exploration

    if ($LOG) { $factor = 10; } #first step goes logarithmically

    my $maxacc = -1;
    my $depth = 0;
    my $C;

    while ($depth < $MAX_DEPTH){
       if ($verbose) {
          COMMON::report($report, "\n=============================================================================================\nLEVEL = $depth :: C-RANGE = [$MIN_C..$MAX_C] :: FACTOR = [".($LOG? "*" : "+")." $factor ]\n=============================================================================================\n"); } 

       my $i = 0;
       $C = $MIN_C;

       while (((sprintf("%.4f", $MAX_C - $C) + 0) >= 0) && ($C >= 0)) {

          if (($depth == 0) and ($i == -1)) { #BUILDING COMPLEMENTARY MODELS - only if they didn't exist
             if ($verbose) { COMMON::report($report, "\n[INIT] BUILDING COMPLEMENTARY MODELS...\n"); }
             if ($unknown) { #train complementary model (one not being tuned) outside the loop
                if (!(-e "$model.$modext.$dirext.$COMMON::MRGEXT")) {
                   SVMTAGGER::learn_known($model, $trainset, $smplset, $smplmap, $rdict, $mode, $modext, $direction, $dirext, $config, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
                   SVMTAGGER::merge_models($model, $modext, $dirext, $smplmap, $unksmplmap, $config, $COMMON::MRGEXT, 0, $report, $verbose);
   	        }
             }
             else {
                if (!(-e "$model.$COMMON::unkext.$modext.$dirext.$COMMON::MRGEXT")) {
                   SVMTAGGER::learn_unknown($model, $trainset, $unksmplset, $unksmplmap, $rdict, $mode, $modext, $direction, $dirext, $config, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
                   SVMTAGGER::merge_models($model, $modext, $dirext, $smplmap, $unksmplmap, $config, $COMMON::MRGEXT, 1, $report, $verbose);
                }
             }
	  }

          $i++;
          if ($verbose) { COMMON::report($report, "\n---------------------------------------------------------------------------------------------\n******************************** level - $depth : ITERATION ".($i - 1)." - C = $C - [M$mode :: $direction]\n---------------------------------------------------------------------------------------------\n\n"); }

	  my $acckn; my $accamb; my $accunk; my $accuracy;

          if ($unknown) {
             $config->{Cu} = $C;
             ($acckn, $accamb, $accunk, $accuracy) = do_CV($config, $mode, $direction, $trainset, $nfolders, $rdict, 1, $remakeFLDS, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
 	  }
          else {
             $config->{Ck} = $C;
             ($acckn, $accamb, $accunk, $accuracy) = do_CV($config, $mode, $direction, $trainset, $nfolders, $rdict, 0, $remakeFLDS, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
	  }
          $remakeFLDS = 0;

          if ($accuracy > $maxacc) { $maxacc = $accuracy; $maxiter = $i; $maxdepth = $depth; $maxC = $C; }

          if ($verbose) {
	     COMMON::report($report, "MAX ACCURACY -> $maxacc :: C-value = $maxC :: depth = $maxdepth :: iter = $maxiter\n");
	  }
          #update on C
          if (($depth == 0) && ($LOG == 1)) { $C = $C * $factor; }
          else { $C = $C + $factor; }
       }

       #update on EXPLORATION INTERVAL
       if (($depth == 0) && ($LOG)) { $MIN_C = $maxC / 2; $MAX_C = $maxC * $factor / 2; }
       else { $MIN_C = $maxC - ($factor / 2); $MAX_C = $maxC + ($factor / 2); }

       #update on EXPLORATION FACTOR
       my $olddepth = $depth;
       $depth++;
       if ($depth < $MAX_DEPTH) {
          $factor = ($MAX_C - $MIN_C) / $nSEGMENTS; 
          if ($verbose) {
 	     COMMON::report($report, "\nJUMPING FROM LEVEL $olddepth INTO LEVEL $depth : C = $maxC : built up on iter = $maxiter\n"); } 
       }
    }
    if ($config->{rmfiles}) { system "rm -f $config->{model}.$COMMON::FOLDEXT*"; }

    return ($maxC, $remakeFLDS);
}

sub find_strategy {
    #description _ returns what strategy should be better evaluated on a certain model
    #param1  _ model type
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unkown words on training)    
    #@return _ tagging strategy

    my $mode = shift;

    return $SVMTAGGER::rSTRATS->{$mode};
}

sub find_mext {
    #description _ returns what file extension is associated to a certain model type
    #param1  _ model type
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unkown words on training)    
    #@return _ tagging strategy

    my $mode = shift;

    return $SVMTAGGER::rMEXTS->{$mode};
}

sub test_SVMT
{
   #description _ responsible for testing a given SVMT model.
   #param1  _ model NAME
   #param2  _ tagging strategy
   #param3  _ model direction
   #param4  _ test set
   #param5  _ output file
   #param6  _ report file
   #param7  _ verbosity [0..3]
   #@return _ ($nhits, $nsamples)

   my $model = shift;
   my $strategy = shift;
   my $direction = shift;
   my $testset = shift;
   my $output_file = shift;
   my $report = shift;
   my $verbose = shift;

   SVMTAGGER::do_SVMT_file($model, 0, 0, 1, $direction, $strategy, "", $testset, $output_file, $verbose);
   my $stats = STATS::do_statistics($model, $testset, $output_file, $verbose);
   my ($acckn, $accamb, $accunk, $accuracy) = STATS::get_accuracies($stats);

   if ($verbose > $COMMON::verbose1) {
      COMMON::report($report, "\n---------------------------------------------------------------------------------------------\n");
   }
   if ($verbose > $COMMON::verbose0) {
      COMMON::report($report, "TEST ACCURACY: $accuracy%\nKNOWN [ $acckn% ] AMBIG.KNOWN [ $accamb% ] UNKNOWN [ $accunk% ]\n");
   }
   if ($verbose > $COMMON::verbose1) {
      COMMON::report($report, "---------------------------------------------------------------------------------------------\n");
   }
   return ($acckn, $accamb, $accunk, $accuracy);
}

sub do_CV
{
    #description _ responsible for learning a given SVMT model.
    #param1  _ configuration hash reference
    #param2  _ model type
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unkown words on training)    
    #param3  _ model direction
    #param4  _ training set
    #param5  _ number of folders
    #param6  _ dictionary object reference
    #param7  _ which models to build (0 -> only known words : 1 -> only unknown words : 2 -> both
    #param8  _ remake CV folders?
    #param9  _ report file
    #param10 _ verbosity [0..3]
    #@return _ (accuracy for known words, accuracy for ambiguous known words,
    #           accuracy for unknown words, overall accuracy)

    my $config = shift;
    my $mode = shift;
    my $direction = shift;
    my $trainset = shift;
    my $nfolders = shift;
    my $rdict = shift;
    my $which = shift;
    my $remakeFLDS = shift;
    my $report = shift;
    my $verbose = shift;
    my $Unihan = shift;
    my $BS = shift;

    if ($verbose > $COMMON::verbose1) {
       COMMON::report($report, "---------------------------------------------------------------------------------------------\nPERFORMING $nfolders-fold CROSS-VALIDATION\non <$trainset>\n---------------------------------------------------------------------------------------------\n");
    }

    my $dirext;
    if ($direction eq $COMMON::lrmode) { $dirext .= "$COMMON::lrext"; }
    elsif ($direction eq $COMMON::rlmode) { $dirext .= "$COMMON::rlext"; }
    my $model = $config->{model};

    if ($mode == $COMMON::mode1) { #COOPERATIVE MODELS M1 and M2 -> build counterpart
       if ((!(-e "$model.$COMMON::M2EXT.$dirext.$COMMON::MRGEXT")) or (!(-e "$model.$COMMON::unkext.$COMMON::M2EXT.$dirext.$COMMON::MRGEXT"))) { #build cooperative models [working together] only if they didn't existed
          SVMTAGGER::build_SVMT($model, $trainset, $rdict, $COMMON::mode2, $direction, $config, 2, 1, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
    }
    elsif ($mode == $COMMON::mode2) {
       if ((!(-e "$model.$COMMON::M1EXT.$dirext.$COMMON::MRGEXT")) or (!(-e "$model.$COMMON::unkext.$COMMON::M1EXT.$dirext.$COMMON::MRGEXT"))) { #build cooperative models [working together] only if not existed
          SVMTAGGER::build_SVMT($model, $trainset, $rdict, $COMMON::mode1, $direction, $config, 2, 1, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       }
    }

    my $strategy = SVMTAGGER::find_strategy($mode);
    my $modext = SVMTAGGER::find_mext($mode);
    my @winsetup = ($config->{wlen}, $config->{wcore});

    my $ACCKN = 0;
    my $ACCAMB = 0;
    my $ACCUNK = 0;
    my $ACCURACY = 0;

    my $folded = $config->{model}.".".$COMMON::FOLDEXT;

    if (($remakeFLDS) or (!(-e $folded))) {  #CREATING FOLDERS
       COMMON::create_folders($trainset, $folded, $nfolders, $report, $verbose);
    }

    my $i = 0;
    while ($i < $nfolders) {
       my $ftrainset = "$folded.$i.$COMMON::trainext";
       my $ftestset = "$folded.$i.$COMMON::testext";
       my $fmodel = $config->{model}.".".$COMMON::FOLDEXT.".".$i;
       COMMON::pick_up_folders($folded, $ftrainset, $ftestset, $i, $report, $verbose); #SELECTING FOLDER(i)

       my $dict = "$fmodel.$COMMON::DICTEXT";             #DICTIONARY

       my $supervised = SVMTAGGER::create_dictionary($ftrainset, $config->{LEX}, $config->{BLEX}, $config->{R}, $config->{Dratio}, $dict, $report, $verbose);

       if ($direction eq $COMMON::rlmode) { #right-to-left  --> reverse input corpus
          COMMON::reverse_file($ftrainset, "$ftrainset.$COMMON::revext");
          if ($config->{rmfiles}) { system "rm -f $ftrainset"; }
          $ftrainset = "$ftrainset.$COMMON::revext";
       }

       #train complementary model (one not being tuned) outside the loop
       my $fwhich = $which;
       if ($fwhich == 1) {
	  if (!(-e "$fmodel.$modext.$dirext.$COMMON::MRGEXT")) { $fwhich = 2; }
       }
       else {
	  if (!(-e "$fmodel.$COMMON::unkext.$modext.$dirext.$COMMON::MRGEXT")) { $fwhich = 2; }
       }

       COMMON::write_list($config->{A0k}, $fmodel.".".$COMMON::A0EXT);
       COMMON::write_list($config->{A1k}, $fmodel.".".$COMMON::A1EXT);
       COMMON::write_list($config->{A2k}, $fmodel.".".$COMMON::A2EXT);
       COMMON::write_list($config->{A3k}, $fmodel.".".$COMMON::A3EXT);
       COMMON::write_list($config->{A4k}, $fmodel.".".$COMMON::A4EXT);
       COMMON::write_list($config->{A0u}, $fmodel.".".$COMMON::A0EXT.".".$COMMON::unkext);
       COMMON::write_list($config->{A1u}, $fmodel.".".$COMMON::A1EXT.".".$COMMON::unkext);
       COMMON::write_list($config->{A2u}, $fmodel.".".$COMMON::A2EXT.".".$COMMON::unkext);
       COMMON::write_list($config->{A3u}, $fmodel.".".$COMMON::A3EXT.".".$COMMON::unkext);
       COMMON::write_list($config->{A4u}, $fmodel.".".$COMMON::A4EXT.".".$COMMON::unkext);
       COMMON::write_list(\@winsetup, $fmodel.".".$COMMON::WINEXT);
       COMMON::write_list(DICTIONARY::find_ambp($dict), $fmodel.".".$COMMON::AMBPEXT);
       COMMON::write_list(DICTIONARY::find_unkp($dict), $fmodel.".".$COMMON::UNKPEXT);

       my $frdict = new DICTIONARY($dict, "$fmodel.$COMMON::AMBPEXT", "$fmodel.$COMMON::UNKPEXT");
       if ($verbose > $COMMON::verbose1) { COMMON::report($report, "DICTIONARY <$dict> [".$frdict->get_nwords." words]\n"); }

       SVMTAGGER::build_SVMT($fmodel, $ftrainset, $frdict, $mode, $direction, $config, $fwhich, ($fwhich != 1), ($fwhich != 0), $report, $verbose,$Unihan,$BS);# have modified  add $Unihan

       my ($acckn, $accamb, $accunk, $accuracy) = test_SVMT($fmodel, $strategy, $direction, $ftestset, "$model.TEST.T$strategy.$direction.$i", $report, $verbose);

       $ACCURACY += $accuracy;
       $ACCKN += $acckn;
       $ACCAMB += $accamb;
       $ACCUNK += $accunk;

       if ($config->{rmfiles}) { system "rm -f $ftrainset";
                                 system "rm -f $ftestset";
                                 system "rm -f $model.TEST.T$strategy.$direction.$i";
       }

       $i++;
    }

    $ACCURACY /= $nfolders;
    $ACCKN /= $nfolders;
    $ACCAMB /= $nfolders;
    $ACCUNK /= $nfolders;

    if ($verbose > $COMMON::verbose1) {
       COMMON::report($report, "\n---------------------------------------------------------------------------------------------\n$nfolders-fold CROSS-VALIDATION\n");
    }
    if ($verbose > $COMMON::verbose0) {
       COMMON::report($report, "OVERALL ACCURACY [Ck = $config->{Ck} :: Cu = $config->{Cu}] : $ACCURACY%\nKNOWN [ $ACCKN% ] AMBIG.KNOWN [ $ACCAMB% ] UNKNOWN [ $ACCUNK% ]\n");
    }
    if ($verbose > $COMMON::verbose1) {
       COMMON::report($report, "---------------------------------------------------------------------------------------------\n\n");
    }

    return ($ACCKN, $ACCAMB, $ACCUNK, $ACCURACY);
}

sub SVMT_learn
{
    #description _ responsible for learning a given SVMT model.
    #param1  _ configuration hash reference
    #param2  _ model type
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unkown words on training)    
    #param3  _ model direction
    #param4  _ C parameter tuning options for KNOWN WORDS
    #          C:begin:end:n_iters:n_segments:[log|nolog]:[V|CV]:CV_n_folders
    #          e.g. C:0.001:10:3:5:log:V
    #          e.g. C:0.001:10:3:5:nolog:V
    #          e.g. C:0.001:10:3:5:log:CV:10
    #param5  _ C parameter tuning options for UNKNOWN WORDS
    #          C:begin:end:n_iters:n_segments:[log|nolog]:[V|CV]:CV_n_folders
    #          e.g. C:0.001:10:3:5:log:V
    #          e.g. C:0.001:10:3:5:nolog:V
    #          e.g. C:0.001:10:3:5:log:CV:10
    #param6  _ test options
    #          T[:CV_n_folders]
    #          e.g T
    #          e.g T:10
    #param7  _ dictionary object reference
    #param8  _ supervised corpus? 0/1
    #param9  _ report file
    #param10 _ verbosity [0..3]

    my $config = shift;
    my $mode = shift;
    my $direction = shift;
    my $CKTopt = shift;
    my $CUTopt = shift;
    my $Topt = shift;
    my $rdict = shift;
    my $supervised = shift;
    my $report = shift;
    my $verbose = shift;
    
    my $Unihan = read_unihan(); #  new added 
    my $BS = read_bs();
    
    if ((!$supervised) and ($mode != $COMMON::mode3)) { die "[mode $mode] SUPERVISED CORPUS NOT AVAILABLE!\n"; }
    if ($mode == $COMMON::mode3) { die "[mode $mode] UNSUPERVISED LEARNING NOT AVAILABLE!\n"; }

    if (!($supervised)) {
       my $eqCs = $rdict->determine_eq_classes($config->{AP}, $config->{Eratio});
       foreach my $e (sort keys %{$eqCs}) {
          print $e, " \t: \t", join(" ", @{$eqCs->{$e}}), "\n";
       }
    }
    #exit;

    if ($verbose) {
       COMMON::report($report, "\n*********************************************************************************************\nBUILDING MODELS... [MODE = $mode :: DIRECTON = $direction]\n*********************************************************************************************\n\n");
    }
    #my $set = $config->{set};       #BIGGER SET
    my $trainset = $config->{trainset};
    my $valset = $config->{valset};
    my $testset = $config->{testset};
    my $model = $config->{model};

    if ($direction eq $COMMON::rlmode) { #right-to-left  --> reverse input corpus
       #system ("tac $trainset > $model.$COMMON::revext");
       COMMON::reverse_file($trainset, "$model.$COMMON::revext");
       $trainset = "$model.$COMMON::revext";
    }

    my $remakeFK = 1;        #remake samples for known words
    my $remakeFU = 1;        #remake samples for unknown words
    my $remakeFLDS = $config->{remakeFLDS};      #remake CV folders

    my @Toptions = split(/;/, $Topt);
    my $NFOLDERS = $Toptions[1];
    my $TTYPE = $Toptions[0];
    my @CKToptions = split(/;/, $CKTopt);
    my @CUToptions = split(/;/, $CUTopt);
    my $doCK = (scalar(@CKToptions) > 2);
    my $doCU = (scalar(@CUToptions) > 2);

    if ((!$doCK) and (scalar(@CKToptions) == 2)) { $config->{Ck} = $CKToptions[1]; }
    if ((!$doCU) and (scalar(@CUToptions) == 2)) { $config->{Cu} = $CUToptions[1]; }

    my $Ck = $config->{Ck};
    my $Cu = $config->{Cu};

    if (($TTYPE eq "T") and ($NFOLDERS > 1)) { $TTYPE = "CVT"; }
    if ($NFOLDERS == 0) { $NFOLDERS = $SVMTAGGER::NFOLDERS; }
    
    if ($doCK or $doCU) { #C PARAMETER TUNING
       my $oldrm = $config->{rmfiles}; $config->{rmfiles} = 0; #remove disabled [to avoid remaking features!]

       if (($TTYPE eq "T") or ($TTYPE eq "")) { #ON VALIDATION SET
	  if ($config->{valset} ne "") {
	     if ($doCK) {
                ($Ck, $remakeFK) = SVMTAGGER::adjust_C($model, $trainset, $rdict, $mode, $direction, $config, $CKTopt, $config->{valset}, 0, $report, $verbose,$Unihan,$BS); # have modified  add $Unihan #TUNE KNOWN WORDS
                $config->{Ck} = $Ck;
	     }
	     if ($doCU) {
                ($Cu, $remakeFU) = SVMTAGGER::adjust_C($model, $trainset, $rdict, $mode, $direction, $config, $CUTopt, $config->{valset}, 1, $report, $verbose,$Unihan,$BS); # have modified  add $Unihan #TUNE UNKNOWN WORDS
                $config->{Cu} = $Cu;
	     }
	  }
          else {
	     if ($verbose) {
 	        COMMON::report($report, "[ERROR] NO VALIDATION SET PROVIDED -> SKIPPING C PARAMETER TUNING!\n");
	     }
	  }
       }
       else { #CVT CROSS-VALIDATION~alike
          if ($doCK) {
             ($Ck, $remakeFLDS) = SVMTAGGER::adjust_C_CV($model, $config->{trainset}, $rdict, $NFOLDERS, $mode, $direction, $config, $CKTopt, 0, $remakeFLDS, $report, $verbose,$Unihan,$BS); # have modified  add $Unihan #TUNE KNOWN WORDS
             $config->{Ck} = $Ck;
	  }
	  if ($doCU) {
             ($Cu, $remakeFLDS) = SVMTAGGER::adjust_C_CV($model, $config->{trainset}, $rdict, $NFOLDERS, $mode, $direction, $config, $CUTopt, 1, $remakeFLDS, $report, $verbose,$Unihan,$BS); # have modified  add $Unihan #TUNE UNKNOWN WORDS
             $config->{Cu} = $Cu;
	  }
       }

       $config->{rmfiles} = $oldrm;
    }

    if ($verbose) { COMMON::report($report, "\n=============================================================================================\n=============================================================================================\n"); }

    if ($TTYPE ne "CVT") { #TRADITIONAL 1-fold
       if ($verbose) {
         COMMON::report($report, "---------------------------------------------------------------------------------------------\n---------------------------------------------------------------------------------------------\nBUILDING MODELS ON WHOLE CORPUS <".$trainset.">\nC PARAMETER [MODE = $mode :: DIRECTON = $direction]:   KNOWN [ C = ".$config->{Ck}." ] :: UNKNOWN [ C = ".$config->{Cu}." ]\n---------------------------------------------------------------------------------------------\n---------------------------------------------------------------------------------------------\n"); 
       }
       SVMTAGGER::build_SVMT($model, $trainset, $rdict, $mode, $direction, $config, 2, $remakeFK, $remakeFU, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       if (($TTYPE eq "T") and ($config->{testset} ne "")) { #ON VALIDATION SET if available
          my $strategy = SVMTAGGER::find_strategy($mode);
	  my ($acckn, $accamb, $accunk, $accuracy) = test_SVMT($model, $strategy, $direction, $config->{testset}, "$model.TEST.T$strategy.$direction", $report, $verbose);
       }
    }
    else { #CROSS-VALIDATION~alike
       if ($verbose) {
         COMMON::report($report, "--------------------------------------------------------------------------------------------\nTRAINING/TESTING MODELS [fair $NFOLDERS-cross-validation]\non <".$config->{trainset}.">\nKNOWN [ C = ".$config->{Ck}." ] :: UNKNOWN [ C = ".$config->{Cu}." ]\n--------------------------------------------------------------------------------------------\n"); 
       }
       my ($acckn, $accamb, $accunk, $accuracy) = do_CV($config, $mode, $direction, $config->{trainset}, $NFOLDERS, $rdict, 2, $remakeFLDS, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
       if ($config->{rmfiles}) { system "rm -f $config->{model}.$COMMON::FOLDEXT*"; }
       if ($verbose) {
         COMMON::report($report, "---------------------------------------------------------------------------------------------\n---------------------------------------------------------------------------------------------\nBUILDING MODELS ON WHOLE CORPUS <".$trainset.">\nC PARAMETER [MODE = $mode :: DIRECTON = $direction]:   KNOWN [ C = ".$config->{Ck}." ] :: UNKNOWN [ C = ".$config->{Cu}." ]\n---------------------------------------------------------------------------------------------\n---------------------------------------------------------------------------------------------\n"); 
       }
       SVMTAGGER::build_SVMT($model, $trainset, $rdict, $mode, $direction, $config, 2, 1, 1, $report, $verbose,$Unihan,$BS);# have modified  add $Unihan
    }

    if (($direction eq $COMMON::rlmode) and ($config->{rmfiles})) {
       if (-e "$model.$COMMON::revext") {
          system "rm -f $model.$COMMON::revext";
       }
    }
}

1;

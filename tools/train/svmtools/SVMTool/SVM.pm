package SVM;

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
use Data::Dumper;
use SVMTool::COMMON;
#use Fcntl;
#use DB_File;

# ---------------- SVM ----------------------------------------
$SVM::KTYPE = "ktype";
$SVM::TL = 0;                    # 0: linear (default)
$SVM::TP = 1;                    # 1: polynomial (s a*b+c)^d
$SVM::TR = 2;                    # 2: radial basis function exp(-gamma ||a-b||^2)
$SVM::TS = 3;                    # 3: sigmoid tanh(s a*b + c)
$SVM::TU = 4;                    # 4: user defined kernel from kernel.h
$SVM::svmseparator = ":";

# -------------------------------------------------------------

# ===================================== SVM ============================================

 sub svm_learn
{
   #description _ calls svm_learn
   #param1 _ MODEL NAME
   #param2 _ SAMPLE SET DSF filename
   #param3 _ TAG
   #param4 _ svm_learn options (parameter C...)
   #param5 _ SVM model file extension
   #param6 _ SVM-light directory (Joachims software) (input)
   #param7 _ REPORT filename               (input/output)
   #param8 _ verbosity

   my $model = shift;
   my $currenttraindsf = shift;
   my $pos = shift;
   my $options = shift;
   my $svmext = shift;
   my $svmdir = shift;
   my $report = shift;
   my $verbose = shift;

   #print "\n\n------\n$svmdir------\n\n";
   if ($svmdir !~ /\/$/) { $svmdir .= "/"; }
   if ($verbose > $COMMON::verbose3) { print "$svmdir"."svm_learn -v 0 $options $currenttraindsf $model.$pos.$svmext\n"; }
   #print "---\n$svmdir"."svm_learn -v 0 $options $currenttraindsf $model.$pos.$svmext\n";
   system("$svmdir"."svm_learn -v 0 $options $currenttraindsf $model.$pos.$svmext >/dev/null") == 0 or die "system failed: $?";
}

#sub svm_classify
#{
#   #description _ calls svm_classify
#   #param1 _ SAMPLE SET DSF filename
#   #param2 _ POS
#   #param3 _ SVM model file extension
#   #param4 _ CLSF classification file extension
#   #param5 _ MODEL directory (where svm models are)
#   #param6 _ SVM-light directory (Joachims software) (input)
#   #param7 _ REPORT filename               (input/output)
#   #param8 _ verbosity

#   my $currenttestdsf = shift;
#   my $pos = shift;
#   my $svmext = shift;
#   my $clsfext = shift;
#   my $modeldir = shift;
#   my $svmdir = shift;
#   my $report = shift;
#   my $verbose = shift;

#   if ($svmdir !~ /\/$/) { $svmdir .= "/"; }
#   my $svmfile;
#   if ($modeldir ne "") { $svmfile = $modeldir."/"; }
#   $pos =~ s/\'/\\\'/g;
#   $pos =~ s/\`/\\\`/g;
#   $pos =~ s/\"/\\\"/g;
#   $pos =~ s/\(/\\\(/g;
#   $pos =~ s/\)/\\\)/g;
#   $svmfile .= $pos.".".$svmext; 
#   if ($verbose > $COMMON::verbose3) { print "$svmdir"."svm_classify -v 0 $currenttestdsf $svmfile $pos.$clsfext\n"; }
#   system("$svmdir"."svm_classify -v 0 $currenttestdsf $svmfile $pos.$clsfext >/dev/null") == 0 or die "system failed: $?";
#}

sub svm_classify_sample_primal
{
   #description _ SVM classify
   #param1 _ sample to classify
   #param2 _ referencia al vector de pesos primal (esparso -> hash)
   #param3 _ coeficiente b primal
   #param4 _ EPSILON threshold for W features relevance  (input)
   #@return _ prediction

   my $sample = shift;
   my $rW = shift;
   my $b = shift;   
   my $epsilon = shift;

   my $sum = 0;
   foreach my $s (@{$sample}) {
      my @sl = split($SVM::svmseparator, $s);
      #if (exists($rW->{$sl[0]})) { $sum += $sl[1] * $rW->{$sl[0]}; }
      if (exists($rW->{$sl[0]})) {
	 #if (abs($rW->{$sl[0]}) >= abs($epsilon)) {
            $sum += $sl[1] * $rW->{$sl[0]};
         #}
      }
   }
   $sum -= $b;
   return $sum;
}

sub svm_classify_sample
{
   #description _ SVM classify 
   #param1 _ sample to classify
   #param2 _ all svm models
   #param3 _ POS
   #@return _ prediction

   my $sample = shift;
   my $rmodels = shift;
   my $pos = shift;   
   
   my $model=$rmodels->{$pos};
   my @svm = @{$model};

   #print "--> ($pos) ";
   #my $pred = svm_classify_sample_on_disk($sample, "/home/usuaris/jgimenez/SVM/MODELS/LINEAR/$pos.$SVMEXT");

   #read svm model
   my $aux1;
   my @aux2;

   shift(@svm);                    # version
   $aux1 = shift(@svm);            # kernel type
   @aux2 = split(/ /, $aux1);
   my $ktype = @aux2[0];
   $aux1 = shift(@svm);            # kernel parameter -d
   @aux2 = split(/ /, $aux1);
   my $kd = @aux2[0];           
   $aux1 = shift(@svm);            # kernel parameter -g
   @aux2 = split(/ /, $aux1);
   my $kg = @aux2[0];
   $aux1 = shift(@svm);            # kernel parameter -s
   @aux2 = split(/ /, $aux1);
   my $ks = @aux2[0];
   $aux1 = shift(@svm);            # kernel parameter -r
   @aux2 = split(/ /, $aux1);
   my $kr = @aux2[0];
   $aux1 = shift(@svm);            # empty kernel parameter -u
   @aux2 = split(/ /, $aux1);
   my $ku = @aux2[0];
   $aux1 = shift(@svm);            # highest feature index
   @aux2 = split(/ /, $aux1);
   my $hfi = @aux2[0];
   $aux1 = shift(@svm);            # number of training documents
   @aux2 = split(/ /, $aux1);
   my $ntd = @aux2[0];
   $aux1 = shift(@svm);            # number of support vectors plus 1 
   @aux2 = split(/ /, $aux1);
   my $nsv = @aux2[0];
   $aux1 = shift(@svm);            # threshold b
   @aux2 = split(/ /, $aux1);
   my $b = @aux2[0];

   #print "KTYPE=$ktype, KD=$kd, KG=$kg, KS=$ks, KR=$kr, HFI=$hfi, NTD=$ntd, NSV=$nsv, b=$b\n";
   #      -t int      -> type of kernel function:
   #                     0: linear (default)
   #                     1: polynomial (s a*b+c)^d
   #                     2: radial basis function exp(-gamma ||a-b||^2)
   #                     3: sigmoid tanh(s a*b + c)
   #                     4: user defined kernel from kernel.h

   my %SMPL;
   my @sl;
   foreach my $s (@{$sample}) {
      @sl = split($SVM::svmseparator, $s);
      $SMPL{$sl[0]} = $sl[1];           # filling out sample hash
   }

   my $sum = 0;
   my @vl;
   while (scalar(@svm) > 0) {
      my $line = shift(@svm);          #each following line is a SV (starting with alpha*y)
      my @entry = split(/ /, $line);
      my $alphay = shift(@entry);
      my $ps = 0;
      foreach my $v (@entry) {
         @vl = split($SVM::svmseparator, $v);
         if (exists($SMPL{$vl[0]})) {
            $ps += ($SMPL{$vl[0]} * $vl[1]);
         }
      }
      if ($ktype == 1) { $ps = ($ps + $kr) ** $kd; }   #polynomial
      $sum += ($alphay * $ps);
   }

   $sum -= $b;

   #print "$sum   VS   $pred\n";

   return $sum;
}

# =================================== MODELS ==========================================

sub read_svm_header
{
   #description _ reads a svm model header
   #param1 _ MODEL filehandle

   my $SVM = shift;

   my ($version, $ktype, $kd, $kg, $ks, $kr, $ku, $hfi, $ntd, $nsv, $b);
   my @aux;
   my $s;

   if (defined($s = $SVM->getline())) { chomp($s); $version = $s; } # version
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $ktype = $aux[0]; } # kernel type
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $kd = $aux[0]; }    # kernel parameter -d
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $kg = $aux[0]; }    # kernel parameter -g
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $ks = $aux[0]; }    # kernel parameter -s
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $kr = $aux[0]; }    # kernel parameter -r
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $ku = $aux[0]; }    # empty kernel parameter -u
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $hfi = $aux[0]; }   # highest feature index
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $ntd = $aux[0]; }   # number of training documents
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $nsv = $aux[0]; }   # number of support vectors plus 1 
   if (defined($s = $SVM->getline())) { @aux = split(/ /, $s); $b = $aux[0]; }     # threshold b

   #print "KTYPE=$ktype, KD=$kd, KG=$kg, KS=$ks, KR=$kr, HFI=$hfi, NTD=$ntd, NSV=$nsv, b=$b\n";

   return ($version, $ktype, $kd, $kg, $ks, $kr, $ku, $hfi, $ntd, $nsv, $b);
}

sub read_models
{
   #description _ reads SVM model files from disk onto memory
   #param1 _ MODEL NAME                                       (input)
   #param2 _ POS list reference                               (input)
   #param3 _ optional extension (used for unknown words SVM)  (input)
   #param4 _ EPSILON/OMEGA threshold for W features relevance (input)
   #param5 _ MAPPING dictionary hash reference                (input)
   #param6 _ VERBOSE (input)
   #@return _ (Weight vector hash reference, Biases hash reference)

   my $model = shift;
   my $rpos = shift;
   my $optext = shift;
   my $epsilon = shift;
   my $rmap = shift;
   my $verbose = shift;

   my $fileW = $model;
   my $fileB = $model;

   if ($optext ne "") {
      $fileW .= ".$optext";
      $fileB .= ".$optext";
      $optext .= ".";
   }

   $fileW .= ".$COMMON::Wext";
   $fileB .= ".$COMMON::Bext";

   my $dir = $model;
   if ($dir =~ /.*\/.*/) { $dir =~ s/\/[^\/]*/\//g; }
   else { $dir = "./"; }
   my ($rW, $rB, $N) = read_primal_models($model, $dir, $rpos, $fileW, $fileB, $optext, $epsilon, $rmap, $verbose);

   return ($rW, $rB);
}


sub read_primal_models
{
   #description _ reads primal models assumed they've been stored in the given directory,
   #              otherwise dual formulation is read and then primal models are written back,
   #              although they might be useless if the kernel type's not linear.
   #param1 _ MODEL NAME                                       (input)
   #param2 _ MODELS directory                                 (input)
   #param3 _ POS list reference
   #param4 _ W filename
   #param5 _ B filename
   #param6 _ optional extension (used for unknown words SVM)  (input)
   #param7 _ EPSILON threshold for W features relevance       (input)
   #param8 _ MAPPING dictionary hash reference                (input)
   #param9 _ VERBOSE (input)
   #@return _ (rW, rB, average W size)

   my $model = shift;
   my $dir = shift;
   my $tagset = shift;
   my $fileW = shift;
   my $fileB = shift;
   my $optext = shift;
   my $epsilon = shift;
   my $rmap = shift;
   my $verbose = shift;

   my ($rW, $rB) = read_dual_models($model, $dir, $tagset, $optext, $epsilon, $rmap, $verbose);
   write_primal_models($rW, $rB, $fileW, $fileB);
   my $ID = 0;
   while ($ID < scalar(@{$tagset})) { # *.SVM won't be longer necessary
      system "rm -f $dir$model.$ID.$optext$COMMON::SVMEXT"; #REMOVE ORIGINAL SVM files
      $ID++;
   }

   my $WM = new IO::File("< $fileW") or die "Couldn't open input file: $fileW\n";
   my $BM = new IO::File("< $fileB") or die "Couldn't open input file: $fileB\n";

   my $n = 0;
   my %W;
   while (defined(my $line = $WM->getline())) {
      my @entry = split(" ", $line);
      my $pos = shift(@entry);
      my %Wpos;
      foreach my $v (@entry) {
	 my @vl = split($SVM::svmseparator, $v);
	 $Wpos{$vl[0]} = $vl[1];
         $n++;
      }
      $W{$pos} = \%Wpos;
   }
   $WM->close();
   
   my $N = 0;
   if (scalar(keys %W) > 0) { $N = $n / scalar(keys %W); }

   my %B;
   while (defined(my $line = $BM->getline())) {
      chomp($line);
      my @entry = split($SVM::svmseparator, $line);
      $B{$entry[0]} = $entry[1];
   }
   $BM->close();

   #system("rm $fileW");
   #system("rm $fileB");

   return (\%W, \%B, $N);
}


sub read_dual_models
{
   #description _ reads SVM model files from disk onto memory (in their dual formulation)
   #param1 _ MODEL NAME                                       (input)
   #param2 _ MODELS directory                                 (input)
   #param3 _ POS list reference                               (input)
   #param4 _ optional extension (used for unknown words SVM)  (input)
   #param5 _ EPSILON threshold for W features relevance       (input)
   #param6 _ MAPPING dictionary hash reference                (input)
   #param7 _ VERBOSE (input)
   #@return _ (Weight vector hash reference, B coefficient hash reference)

   my $model = shift;
   my $dir = shift;
   my $tagset = shift;
   my $optext = shift;
   my $epsilon = shift;
   my $rmap = shift;
   my $verbose = shift;

   my %W;
   my %B;

   my $ID = 0;
   while ($ID < scalar(@{$tagset})) {
       my $tag = $tagset->[$ID];

       if ($verbose > $COMMON::verbose1) { print "$tag.."; }
       my $SVM = new IO::File("< $dir$model.$ID.$optext$COMMON::SVMEXT") or die "Couldn't open input file: $dir$model.$ID.$optext$COMMON::SVMEXT\n";

       my ($version, $ktype, $kd, $kg, $ks, $kr, $ku, $hfi, $ntd, $nsv, $b) = read_svm_header($SVM);
       my @model = ($version, $ktype, $kd, $kg, $ks, $kr, $ku, $hfi, $ntd, $nsv, $b);

       # ----------------------------------------------------
       $B{$tag} = $b;
       # ----------------------------------------------------

       my %Wpos;
       while (defined(my $s = $SVM->getline())) {          
	  chomp($s);
	  push(@model, $s);
          my @entry = split(/ /, $s);
          my $alphay = shift(@entry);
          foreach my $v (@entry) {
             my @vl = split($SVM::svmseparator, $v);
	     if (exists($Wpos{$vl[0]})) {
	        $Wpos{$vl[0]} += $alphay * $vl[1];
	     }
	     else {
	        $Wpos{$vl[0]} = $alphay * $vl[1];
	     }
	  }
       }

       if ($epsilon != 0) {
          # now's the time for filtering irrelevant features out
          my %Wposf;
          foreach my $w (keys %Wpos) {
	     if (abs($Wpos{$w}) >= abs($epsilon)) { #feature is rellevant
                $Wposf{$w} = $Wpos{$w};  
  	     }
          }
          $W{$tag} = \%Wposf;
       }
       else { #epsilon == 0 --> don't filter
          $W{$tag} = \%Wpos;
       }

       # ----------------------------------------------------

       $SVM->close();
       $ID++;
   }
   if ($verbose > $COMMON::verbose1) { print "[DONE]\n"; }

   return (\%W, \%B);
}


sub write_primal_models
{
   #description _ writes primal models onto the given directory
   #param1 _ referencia al hash de vectores de pesos 
   #param2 _ referencia al hash de coeficientes b
   #param3 _ W filename
   #param4 _ B filename

   my $rW = shift;
   my $rB = shift;
   my $fileW = shift;
   my $fileB = shift;

   my $WM = new IO::File("> $fileW") or die "Couldn't open output file: $fileW\n";
   foreach my $p (sort keys %{$rW}) {
       my $entry = "$p ";
       foreach my $v (sort keys %{$rW->{$p}}) {
          $entry .= $v.$SVM::svmseparator.$rW->{$p}->{$v}." ";
       }
       chop($entry);
       print $WM "$entry\n";
   }
   $WM->close();

   my $BM = new IO::File("> $fileB") or die "Couldn't open output file: $fileB\n";
   foreach my $p (sort keys %{$rB}) {
       print $BM $p, $SVM::svmseparator, $rB->{$p}, "\n";
   }

   $BM->close();
}

sub write_merged_models
{
   #description _ writes merged models
   #param1 _ referencia al hash que contiene el MERGED MODEL
   #param2 _ Biases hash reference
   #param3 _ config structure
   #param4 _ file name
   #param5 _ (1/0) unknown/known model

   my $rM = shift;
   my $rB = shift;
   my $config = shift;
   my $fileM = shift;
   my $unknown = shift;

   my $FM = new IO::File("> $fileM") or die "Couldn't open output file: $fileM\n";

   print $FM "# $COMMON::appname v$COMMON::appversion MERGED PRIMAL MODEL\n# SLIDING WINDOW: length [$config->{wlen}] :: core [$config->{wcore}]\n# FEATURE FILTERING: min frequency [$config->{fmin}] :: max mapping size [$config->{maxmapsize}]\n# C-PARAMETER: ".($unknown? $config->{Cu} : $config->{Ck})."\n# =========================================================================\n";

   my @bline;
   foreach my $bias (sort keys %{$rB}) {
      push(@bline, $bias.$SVM::svmseparator.$rB->{$bias});
   }
   print $FM "BIASES ".join(" ", @bline)."\n";

   foreach my $att (sort keys %{$rM}) {
      my @wline;
      foreach my $pos (sort keys %{$rM->{$att}}) {
	 push(@wline, $pos.$SVM::svmseparator.$rM->{$att}->{$pos});
      }
      print $FM "$att ", join(" ", @wline), "\n";
   }

   $FM->close();
}

#sub write_merged_models_BDB
#{
   #description _ writes merged models
   #param1 _ referencia al hash que contiene el MERGED MODEL
   #param2 _ Biases hash reference
   #param3 _ file name

#   my $rM = shift;
#   my $rB = shift;
#   my $fileM = shift;

#   my %h;
#   tie(%h,'DB_File', $fileM,  O_RDWR|O_CREAT , 0644 , $DB_HASH) or die "can't tie DB_File $! $fileM";   

#   my @bline;
#   foreach my $bias (sort keys %{$rB}) {
#      push(@bline, $bias.$SVM::svmseparator.$rB->{$bias});
#   }
#   $h{"BIASES"} = join(" ", @bline)."\n";

#   foreach my $att (sort keys %{$rM}) {
#      my @wline;
#      foreach my $pos (sort keys %{$rM->{$att}}) {
#   	 push(@wline, $pos.$SVM::svmseparator.$rM->{$att}->{$pos});
#      }
#      $h{$att} = join(" ", @wline);
#   }

#   untie %h;                          # Flush and close the dbm file
#}


sub read_B
{
   #description _ reads only B coefficients from an SVM model
   #param1 _ file name                               (input)
   #@return _ B coefficient hash reference

   my $fB = shift;

   my $BM = new IO::File("< $fB") or die "Couldn't open input file: $fB\n";
   my %B;

   while (defined(my $line = $BM->getline())) {
      my @entry = split($SVM::svmseparator, $line);
      $B{$entry[0]} = $entry[1];
   }

   $BM->close();

   return (\%B);
}

sub read_merged_models
{
   #description _ reads merged models
   #param1 _ file name
   #param2 _ EPSILON threshold for W features relevance       (input)
   #param3 _ VERBOSE (input)
   #@return _ referencia al hash que contendra el MERGED MODEL

   my $fileM = shift;
   my $epsilon = shift;
   my $verbose = shift;

   my $FM = new IO::File("< $fileM") or die "Couldn't open input file: $fileM\n";
   my %B;
   my %M;
   my $N;
   my $Nout;

   while (defined(my $line = $FM->getline())) {
      if (($line ne "") and ($line !~ /^\#.*/)) {
         if ($line =~ /^BIASES /) {
            my @biases = split(" ", $line);
            my $i = 1;
            while ($i < scalar(@biases)) {
               my @elem = split($SVM::svmseparator, $biases[$i]);
               $B{$elem[0]} = $elem[1];
               $i++;
            }
         }
         else {
            my @entry = split(" ", $line);
            my $att = shift(@entry);
            my %Matt;
            foreach my $v (@entry) {
	       my @vl = split($SVM::svmseparator, $v);
               if (abs($vl[1]) >= abs($epsilon)) { # now's the time for filtering irrelevant features out
                  $Matt{$vl[0]} = $vl[1]; #feature is rellevant
               }
               else { $Nout++; }
               $N++;
            }
            $M{$att} = \%Matt;
         }
      }
   }

   $FM->close();

   if (($verbose) and ($epsilon > 0)) { print STDERR "[filtering weights on +/- $epsilon] --> [", $Nout+0, " / $N] = ", sprintf("%.4f", $Nout/$N * 100) + 0, " % discarded\n"; }

   #print STDERR "WRITING BDB MERGED MODELS < $fileM.2 >\n";
   #SVM::write_merged_models_BDB(\%M, \%B, "$fileM.2");

   return (\%M, \%B, $N, $Nout);
}

#sub read_merged_models_BDB
#{
   #description _ reads merged models
   #param1 _ file name
   #param2 _ EPSILON threshold for W features relevance       (input)
   #param3 _ VERBOSE (input)
   #@return _ referencia al hash que contendra el MERGED MODEL

#   my $fileM = shift;
#   my $epsilon = shift;
#   my $verbose = shift;

#   my %B;
#   my %M;
#   my $N;
#   my $Nout;

#   my %h;
#   tie(%h,'DB_File', $fileM,  O_RDWR|O_CREAT , 0644 , $DB_HASH) or die "can't tie DB_File $! $fileM";   

#   foreach my $k (keys %h) {
#      if ($k eq "BIASES") { #biases
#         my @biases = split(" ", $h{$k});
#         my $i = 0;
#         while ($i < scalar(@biases)) {
#            my @elem = split($SVM::svmseparator, $biases[$i]);
#            $B{$elem[0]} = $elem[1];
#            $i++;
#         }
#      }
#      else { #weights  
#         my @entry = split(" ", $h{$k});
#         my %Matt;
#         foreach my $v (@entry) {
#            my @vl = split($SVM::svmseparator, $v);
#            if (abs($vl[1]) >= abs($epsilon)) { # now's the time for filtering irrelevant features out
#               $Matt{$vl[0]} = $vl[1]; #feature is rellevant
#            }
#            else { $Nout++; }
#            $N++;
#         }
#         $M{$k} = \%Matt;
#      }
#   }

#   if (($verbose) and ($epsilon > 0)) { print STDERR "[filtering weights on +/- $epsilon] --> [", $Nout+0, " / $N] = ", sprintf("%.4f", $Nout/$N * 100) + 0, " % discarded\n"; }

#   untie %h;                          # Flush and close the dbm file

#   return (\%M, \%B, $N, $Nout);
#}

sub merge_models
{
   #description _ merges mapping and primal weights onto a new model.
   #param1 _ reverse mapping hash reference
   #param2 _ referencia al hash de vectores de pesos primal
   #@return _ merged model hash reference

   my $rmap = shift;
   my $rW = shift;

   my %MODEL;

   foreach my $pos (keys %{$rW}) {
      foreach my $att2 (keys %{$rW->{$pos}}) {
	 $MODEL{$rmap->{$att2}}->{$pos} = $rW->{$pos}->{$att2};
      }
   }

   return(\%MODEL);
}

sub write_mem_sample
{
    #description _ responsible for writing down a sample onto memory.
    #
    #param1 _ sample attributes hash reference
    #@return value : attribute list

    my $rattribs = shift;

    my @output;

    foreach my $sample (keys %{$rattribs}) { #sorting is not necessary
       push(@output, $sample.$COMMON::attvalseparator.$rattribs->{$sample});
    }

    return \@output;
}

sub write_sample
{
    #description _ responsible for writing the sample onto the corresponding files.
    #param1 _ label tag (word:pos)
    #param2 _ sample attributes hash reference
    #param3 _ sample file handle

    my $label = shift;
    my $rattribs = shift;
    my $SMPLSET = shift;

    if (defined($SMPLSET)) {
       my @output;
       push(@output, $label);

       foreach my $sample (sort keys %{$rattribs}) { #sorting is mandatory
   	   push(@output, $sample.$COMMON::attvalseparator.$rattribs->{$sample});
       }

       #actually writing the sample onto the file
       print $SMPLSET join($COMMON::pairseparator, @output)."\n";
    }
}

1;


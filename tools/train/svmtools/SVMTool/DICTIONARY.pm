package DICTIONARY;

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

# ======================================================================================
# ================================= DICTIONARY METHODS =================================
# ======================================================================================

sub new
{
    #description _ creates, given a dictionary file name, the corresponding
    #              instance of a dictionary object.
    #param1 _ dictionary file name
    #param2 _ ambiguous pos list file
    #param3 _ unknown pos list file

    my $class = shift;
    my $dict = shift;
    my $fambp = shift;
    my $funkp = shift;

    my $rdict = {};

    $rdict = { "nwords" => 0, "dict" => {}, rAP => 0, rUP => 0, rAMBP => 0, rUNKP => 0, AVGF => 0 };

    bless $rdict, $class;

    if ($dict ne "0") {
       my $F = 0;
       my $N = 0;
       my $DICTIONARY = new IO::File("< $dict") or die "Couldn't open input file: $dict\n";
       while (defined(my $line = $DICTIONARY->getline())) {
          chomp($line);
          my @entrada = split(/ +/, $line);
          my $word = $entrada[0];
          $F += $entrada[1];
          $N++;
          shift @entrada;
          my $newline = join($COMMON::d_separator, @entrada);
          $rdict->write_entry($word, $newline);
       }
       $DICTIONARY->close();
       $rdict->{AVGF} = $F / $N;  #average_frequency
    }

    if (-e $fambp) {
       $rdict->{rAP} = COMMON::read_list($fambp);
       $rdict->{rAMBP} = COMMON::do_hash($rdict->{rAP}, "");
    }
    else {
       $rdict->{rAP} = ();
       $rdict->{rAMBP} = {};
    }

    if (-e $funkp) {
       $rdict->{rUP} = COMMON::read_list($funkp);
       $rdict->{rUNKP} = COMMON::do_hash($rdict->{rUP}, "");
    }
    else {
       $rdict->{rUP} = ();
       $rdict->{rUNKP} = {};
    }

    return $rdict;
}

sub write_entry
{
    #description _ writes the given entry for the given word onto the given dictionary.
    #param1 _ dictionary reference (implicit)
    #param2 _ word
    #param3 _ entry

    my $rdict = shift;
    my $word = shift;
    my $entry = shift;

    my @l = split($COMMON::d_separator, $entry);
    $rdict->{"dict"}->{$word} = \@l;
    $rdict->{"nwords"}++;
}

sub read_entry
{
    #description _ given a word returns its dictionary entry
    #param1 _ dictionary object reference (implicit)
    #param2 _ word to retrieve

    my $rdict = shift;
    my $word = shift;

    if (defined($rdict->{"dict"}->{$word})) { return(@{$rdict->{"dict"}->{$word}}); }
    else { return (); }
}

sub get_potser_old
{
    #description _ given a word returns its dictionary "potsers".
    #param1 _ dictionary object reference (implicit)
    #param2 _ word

    my $rdict = shift;
    my $word = shift;

    my @entry = $rdict->read_entry($word);
    my $rpotser=[];

    if (defined(@entry)) {
       my $n = $entry[1]; 
       my $i = 2;

       while ($i <= ($n * 2) )
       {
          push(@{$rpotser}, $entry[$i]);
          $i+=2;
       }
    }

    return($rpotser);
}


sub get_potser
{
    #description _ given a word returns its dictionary "potsers".
    #              (modified to be robust on dictionary errors)
    #param1 _ dictionary object reference (implicit)
    #param2 _ word

    my $rdict = shift;
    my $word = shift;

    my @entry = $rdict->read_entry($word);
    my @ps;

    my $n = $entry[1]; 
    my $i = 2;

    my $stop = 0;
    while (($i <= ($n * 2)) && ($stop == 0)) {
       if (exists($rdict->{rAMBP}->{$entry[$i]})) { #check only possibly ambiguous POS
          push(@ps, $entry[$i]);
       }
       else { #is it a DICTIONARY ERROR?
          my $mft = $rdict->get_mft($word);
          my $pos = $entry[$i];
          if ($entry[$i] eq $mft) { #ERROR ~ to 27249 3 IN 2 JJ 1 TO 27246
             @ps = ($mft);
             $stop = 1;
          }        
       }
       $i+=2;
    }

    return(\@ps);
}

sub get_real_potser
{
    #description _ given a word returns its dictionary "potsers".
    #              (modified to be robust on dictionary errors)
    #param1 _ dictionary object reference (implicit)
    #param2 _ word

    my $rdict = shift;
    my $word = shift;

    my @entry = $rdict->read_entry($word);
    my @ps;

    if ((scalar(@entry) > 0) and ($entry[1] != 0)) { #KNOWN WORD
       my $n = $entry[1]; 
       my $i = 2;

       my $stop = 0;
       while (($i <= ($n * 2)) && ($stop == 0)) {
          if (exists($rdict->{rAMBP}->{$entry[$i]})) { #check only possibly ambiguous POS
             push(@ps, $entry[$i]);
          }
          else { #is it a DICTIONARY ERROR?
             my $mft = $rdict->get_mft($word);
             my $pos = $entry[$i];
             if ($entry[$i] eq $mft) { #ERROR ~ to 27249 3 IN 2 JJ 1 TO 27246
	        @ps = ($mft);
                $stop = 1;
             }        
          }
          $i+=2;
       }

       return(\@ps);
    }
    else { #UNKNOWN WORD
       if ($word =~ /^[0-9]+$/) { return $rdict->get_real_potser("\@CARD"); }
       elsif ($word =~ /^[0-9]+[\.\,\!\?:]+$/) { return $rdict->get_real_potser("\@CARDPUNCT"); }
       elsif ($word =~ /^[0-9]+[:\.\,\/\\\-][0-9\.\,\-\\\/]+$/) { return $rdict->get_real_potser("\@CARDSEPS"); }
       elsif ($word =~ /^[0-9]+[^0-9]+.*$/) { return $rdict->get_real_potser("\@CARDSUFFIX"); }
       else { return $rdict->{rUP}; }
    }
}

sub get_mft
{
    #description _ given a word returns the most frequent tag.
    #param1 _ dictionary object reference (implicit)
    #param2 _ word

    my $rdict = shift;
    my $word = shift;

    my @entry = $rdict->read_entry($word);

    if (scalar(@entry) == 0) { return ""; }

    my $mft;
    my $n = $entry[1]; 
    my $i = 2;
    my $max = -1;
    while ($i <= ($n * 2) )
    {
       if ($entry[$i+1] > $max) { $mft = $entry[$i]; $max = $entry[$i+1]; }
       $i+=2;
    }

    return($mft);
}

sub frequent_word
{
    #description _ returns true only if the window core word is a reasonably frequent one
    #              (modified for dictionary error robustness)
    #param1 _ dictionary object reference (implicit)
    #param2 _ candidate word

    my $rdict = shift;
    my $word = shift;

    my @entrada = $rdict->read_entry($word);

    if ($entrada[0] > $rdict->{AVGF}) { return 1; }
    else { return 0; }
}

sub ambiguous_word
{
    #description _ returns true only if the window core word is a POS ambiguous one
    #              (modified for dictionary error robustness)
    #param1 _ dictionary object reference (implicit)
    #param2 _ candidate word

    my $rdict = shift;
    my $word = shift;

    my $mft = $rdict->get_mft($word);
    my @entrada = $rdict->read_entry($word);

    if (exists($rdict->{rAMBP}->{$mft})) { return ($entrada[1] > 1); }
    else { return 0; }
}

sub unknown_word
{
    #description _ returns true only if the window core word is an unknown one
    #              (not in the dictionary)
    #param1 _ dictionary object reference (implicit)
    #param2 _ candidate word

    my $rdict = shift;
    my $word = shift;

    return ((!(exists($rdict->{"dict"}->{$word}))) or ($rdict->{"dict"}->{$word}->[1] == 0));
}

sub get_nwords
{
    #description _ returns the number of words in the dictionary
    #param1 _ dictionary object reference (implicit)

    my $rdict = shift;

    return($rdict->{nwords});
}


sub find_ambp
{
   #description _ finds ambiguous parts-of-speech, given a generated dictionary
   #              (trick: look up ambiguous entries)
   #param1 _ DICTIONARY filename

   my $dict = shift;

   my %AP;
   my $rdict = new DICTIONARY($dict, "", "");

   foreach my $word (keys %{$rdict->{"dict"}}) {
      my @entry = $rdict->read_entry($word);
      if ($entry[1] > 1) { #word is ambiguous
          my $i = 2;
          while ($i <= ($entry[1] * 2) )
          {
   	     $AP{$entry[$i]} = 1;
             $i+=2;
          }
      }
   }

   my @AP;
   foreach my $pos (keys %AP) {
      push(@AP, $pos);
   }

   my @SAP = sort @AP;

   return (\@SAP);
}

sub find_unkp
{
   #description _ finds open-class parts-of-speech, given a generated dictionary
   #              (trick: look up words appearing just once)
   #param1 _ DICTIONARY filename

   my $dict = shift;

   my %UP;
   my $rdict = new DICTIONARY($dict, "", "");

   foreach my $word (keys %{$rdict->{"dict"}}) {
      my @entry = $rdict->read_entry($word);
      if ($entry[0] == 1) { #word appears just once
	 $UP{$entry[2]} = 1;
      }
   }

   my @UP;
   foreach my $pos (keys %UP) {
      push(@UP, $pos);
   }

   my @SUP = sort @UP;

   return (\@SUP);
}

sub determine_eq_classes
{
   #description _ finds classes of equivalence for all parts-of-speech in a given list;
   #              a class of equivalence groups classes among which ambiguity conflicts arise.
   #param1  _ dictionary object reference (implicit)
   #param2  _ part-of-speech list reference
   #param3  _ Eratio (to compute classes of equivalence in unsupervised mode)  (input)
   #@return _ classes of equivalence hash reference

   my $rdict = shift;
   my $rpos = shift;
   my $Eratio = shift;

   my %reqC;
   foreach my $pos (@{$rpos}) { #initialize classes of equivalence
      my %auxh;
      #foreach my $pos (@{$rpos}) {
      #   $auxh{$pos} = 0;
      #}
      $reqC{$pos} = \%auxh;
   }

   #my $M = 0; my $P = 0;
   #foreach my $word (keys %{$rdict->{"dict"}}) {
   #   my @entry = $rdict->read_entry($word);
   #   if ($entry[1] > 1) { $M += $entry[0]; $P++; }
   #}
   #print "#AMBWORDS = $M :: DISTINCT = $P :: AVG = ", $M/$P, "\n";
   #my $AVG = $M / $P;

   foreach my $word (keys %{$rdict->{"dict"}}) {
      my @entry = $rdict->read_entry($word);

      if ($word !~ /^\@CARD.*/) {
         if ($entry[1] > 1) { #word is ambiguous
            my $i = 2;
            while ($i <= ($entry[1] * 2) ) {
	       my $j = 2;
               while ($j <= ($entry[1] * 2) ) {
		  if ($i != $j) {
                     $reqC{$entry[$i]}->{$entry[$j]}++;
                  }
                  $j+=2;
	       }
               $i+=2;
            }
	 }
      }
   }

if (0) {
   foreach my $pos (sort keys %reqC) { #CLASS FILTERING
      my $N;
      my %aux;
      foreach my $v (keys %{$reqC{$pos}}) { $N += $reqC{$pos}->{$v}; }
      my $Nratio = $N / scalar(keys %{$reqC{$pos}}) / 3;
      foreach my $v (keys %{$reqC{$pos}}) {
         #print "$pos --> [$v] :: ", $reqC{$pos}->{$v}, " [", $Nratio, "]";
         if ($reqC{$pos}->{$v} >= $Nratio) { #CLASS FILTERING
            #push(@aux, $v);
            $aux{$v} = $reqC{$pos}->{$v};
            #print " ***";
	 }
         #print "\n";
      }
      $reqC{$pos} = \%aux;
   }
}

   foreach my $pos (sort keys %reqC) { #CLASS RE-BALANCING  NEG(X,Y) -> POS(Y,X)
      foreach my $v (keys %{$reqC{$pos}}) {
         $reqC{$v}->{$pos} = $reqC{$pos}->{$v};
      }
   }

   foreach my $pos (sort keys %reqC) { # output
      my @aux;
      foreach my $v (keys %{$reqC{$pos}}) { push(@aux, $v); }
      my @aux2 = sort @aux;
      $reqC{$pos} = \@aux2;
   }

   return \%reqC;
}

sub tag
{
    #description _ if the given word is unambiguous returns its associated tag,
    #              otherwise returns 'emptypos'
    #param1 _ dictionary object reference (implicit)
    #param2 _ word

    my $rdict = shift;
    my $word = shift;

    my @entrada = $rdict->read_entry($word);

    if ($entrada[1] > 1) { return $COMMON::unkpos; }
    else { return $rdict->get_mft($word); }
}

1;

package STATS;

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
use SVMTool::DICTIONARY;

# ------------------------------------------------------------------------------------------
# COMPUTE STATISTICS

sub do_statistics
{
    #description _ compute some figures, percentages...
    #param1 _ model
    #param2 _ input filename
    #param3 _ output filename
    #param4 _ verbosity (0..3)
    #@return _ stat structure reference

    my $model = shift;
    my $input = shift;
    my $output = shift;
    my $verbose = shift;

    if ($verbose > $COMMON::verbose1) { print "EVALUATING <$output> vs. <$input> on model <$model>...\n"; }

    my $dict = $model.".".$COMMON::DICTEXT;
    my $fambp = $model.".".$COMMON::AMBPEXT;
    my $funkp = $model.".".$COMMON::UNKPEXT;

    my $DIE = 0;

    my $INPUT = new IO::File("< $input") or die "Couldn't open input file: $input\n";
    my $OUTPUT = new IO::File("< $output") or $DIE = 1;

    if ($DIE) { print "\n"; die "Couldn't open input file: $output\n"; }

    #dictionary generation
    my $rdict = new DICTIONARY($dict, $fambp, $funkp);

    my $nsamples = 0;
    my $unknown = 0;
    my $ambiguous = 0;
    my $nhits = 0;
    my $mfthits = 0;
    my $nambhits = 0;
    my $nunkhits = 0;
    my $avgamb = 0;
    my $hit = 0;
    my %ambN;
    my %ambK;
    my %ambP;

    while (defined(my $in = $INPUT->getline()) and defined(my $out = $OUTPUT->getline())) {
       chomp ($in);
       chomp ($out);
       my @linein = split(/$COMMON::in_valseparator/, $in);
       my @lineout = split(/$COMMON::out_valseparator/, $out);
       my $wordin = $linein[0];
       my $wordout = $lineout[0];
       my $posin = $linein[1];
       my $posout = $lineout[1];

       if ((($linein[0] ne $COMMON::IGNORE) and ($linein[0] ne "")) and (($lineout[0] ne $COMMON::IGNORE) and ($lineout[0] ne ""))) {

          $nsamples++;
          if ($verbose > $COMMON::verbose2) { COMMON::show_progress($nsamples, $COMMON::progress1, $COMMON::progress2); }

          if ($posin eq $posout) { $nhits++; $hit = 1; } #HIT
          else { $hit = 0; }

          my $posmft = $rdict->get_mft($wordin);

          if ($rdict->unknown_word($wordin)) { #UNKNOWN WORD
             $unknown++;
             if ($hit) { $nunkhits++; }
          }
          else { #KNOWN WORD
             if ($posmft eq $posin) { $mfthits++; }
             if ($rdict->ambiguous_word($wordin)) { #ambiguous
                $ambiguous++;
                if ($hit) { $nambhits++; }
             }
          }

          my $rpotser = $rdict->get_real_potser($wordin);
          $avgamb += scalar(@{$rpotser});

          #AMBIGUITY LEVEL
          if (exists($ambN{scalar(@{$rpotser})})) {
             $ambN{scalar(@{$rpotser})}[0]++;
	     if ($hit) { $ambN{scalar(@{$rpotser})}[1]++; }
             if ($posmft eq $posin) { $ambN{scalar(@{$rpotser})}[2]++; }
	  }
          else {
	     my @laux = (1, 0, 0);
             if ($hit) { $laux[1]++; }
             if ($posmft eq $posin) { $laux[2]++; }
             $ambN{scalar(@{$rpotser})} = \@laux;
          }
          
          #AMBIGUITY KIND
          my $potser = join($COMMON::innerseparator, @{$rpotser});
          if (exists($ambK{$potser})) {
             $ambK{$potser}[0]++;
	     if ($hit) { $ambK{$potser}[1]++; }
             if ($posmft eq $posin) { $ambK{$potser}[2]++; }
	  }
          else {
	     my @laux = (1, 0, 0);
             if ($hit) { $laux[1]++; }
             if ($posmft eq $posin) { $laux[2]++; }
             $ambK{$potser} = \@laux;
	  }

          #POS
          if (exists($ambP{$posin})) {
             $ambP{$posin}[0]++;
	     if ($hit) { $ambP{$posin}[1]++; }
             if ($posmft eq $posin) { $ambP{$posin}[2]++; }
	  }
          else {
	     my @laux = (1, 0, 0);
             if ($hit) { $laux[1]++; }
             if ($posmft eq $posin) { $laux[2]++; }
             $ambP{$posin} = \@laux;
	  }
       }
    }
    if ($verbose > $COMMON::verbose2) { print STDERR "...$nsamples tokens [DONE]\n"; }

    $INPUT->close();
    $OUTPUT->close();

    my %stats;
    $stats{nsamples} = $nsamples;
    $stats{namb} = $ambiguous;
    $stats{nunk} = $unknown;
    $stats{nhits} = $nhits;
    $stats{nambhits} = $nambhits;
    $stats{nunkhits} = $nunkhits;
    $stats{mfthits} = $mfthits;
    $stats{avgamb} = ($nsamples > 0)? $avgamb / $nsamples : 0;

    $stats{ambN} = \%ambN;
    $stats{ambK} = \%ambK;
    $stats{ambP} = \%ambP;

    return(\%stats);
}

# ------------------------------------------------------------------------------------------
# PRINT STDOUT

sub print_stats_header
{
    #description_ prints header of statistics
    #param1 _ statistics hash reference

    my $stats = shift;

    my $nsamples = $stats->{nsamples};
    my $ambiguous = $stats->{namb};
    my $unknown = $stats->{nunk};
    my $nhits = $stats->{nhits};
    my $nambhits = $stats->{nambhits};
    my $nunkhits = $stats->{nunkhits};
    my $mfthits = $stats->{mfthits};
    my $avgamb = $stats->{avgamb};

    my $ambpc = ($nsamples > 0)? $ambiguous / $nsamples * 100 : 0;
    my $mftacc = ($nsamples > 0)? $mfthits / $nsamples * 100 : 0;
    my $knownpc = ($nsamples > 0)? ($nsamples - $unknown) / $nsamples * 100 : 0;
    my $unknownpc = ($nsamples > 0)? $unknown / $nsamples * 100 : 0;

    print "* ================= TAGGING SUMMARY =======================================================\n";
    printf STDOUT "#TOKENS           = %s\n", $nsamples;
    printf STDOUT "AVERAGE_AMBIGUITY = %.4f tags per token\n", $avgamb + 0;
    print "* -----------------------------------------------------------------------------------------\n";
    my $known = $nsamples - $unknown;
    printf STDOUT "#KNOWN            = %7.4f%% --> %16s / %-16s\n", $knownpc + 0, $known, $nsamples;
    printf STDOUT "#UNKNOWN          = %7.4f%% --> %16s / %-16s\n", $unknownpc + 0, $unknown, $nsamples;
    printf STDOUT "#AMBIGUOUS        = %7.4f%% --> %16s / %-16s\n", $ambpc + 0, $ambiguous, $nsamples;
    printf STDOUT "#MFT baseline     = %7.4f%% --> %16s / %-16s\n", $mftacc + 0, $mfthits, $nsamples;
}

sub print_stats_overall
{
    #description_ prints header of statistics
    #param1 _ statistics hash reference

    my $stats = shift;

    my $nsamples = $stats->{nsamples};
    my $nhits = $stats->{nhits};
    my $mfthits = $stats->{mfthits};

    print "* ================= OVERALL ACCURACY ======================================================\n";
    printf STDOUT "%16s %16s  %16s  %16s\n", "HITS", "TRIALS", "ACCURACY", "MFT";
    print "* -----------------------------------------------------------------------------------------\n";
    printf STDOUT "%16d %16d %16.4f%% %16.4f%%\n", $nhits, $nsamples, ($nsamples > 0)? ($nhits / $nsamples) * 100 : 0, ($nsamples > 0)? ($mfthits / $nsamples) * 100 : 0;
    print "* =========================================================================================\n";
}

sub print_stats_ambU
{
    #description_ prints accuracy comparing known vs. unkown tokens
    #param1 _ statistics hash reference

    my $stats = shift;

    my $ambh = $stats->{ambN};

    my $nsamples = $stats->{nsamples};
    my $ambiguous = $stats->{namb};
    my $unknown = $stats->{nunk};
    my $nhits = $stats->{nhits};
    my $nambhits = $stats->{nambhits};
    my $nunkhits = $stats->{nunkhits};
    my $mfthits = $stats->{mfthits};

    print "* ================= KNOWN vs UNKNOWN TOKENS ===============================================\n";
    printf STDOUT "%16s %16s %16s\n", "HITS", "TRIALS", "ACCURACY";
    print "* -----------------------------------------------------------------------------------------\n";
    print "* ======= known ===========================================================================\n";
    printf STDOUT "%16d %16d %10.4f%%\n", $nhits - $nunkhits, $nsamples - $unknown, (($nsamples - $unknown) > 0)? (($nhits - $nunkhits) / ($nsamples - $unknown)) * 100 : 0;
    print "-------- known unambiguous tokens ---------------------------------------------------------\n";
    printf STDOUT "%16d %16d %10.4f%%\n", $nhits - $nunkhits - $nambhits, $nsamples - $unknown - $ambiguous, (($nsamples - $unknown - $ambiguous) > 0)? (($nhits - $nunkhits - $nambhits) / ($nsamples - $unknown - $ambiguous)) * 100 : 0;
    print "-------- known ambiguous tokens -----------------------------------------------------------\n";
    printf STDOUT "%16d %16d %10.4f%%\n", $nambhits, $ambiguous, ($ambiguous > 0)? ($nambhits / $ambiguous) * 100 : 0;
    print "* ======= unknown =========================================================================\n";
    printf STDOUT "%16d %16d %10.4f%%\n", $nunkhits, $unknown, ($unknown > 0)? ($nunkhits / $unknown) * 100 : 0;
    print "* =========================================================================================\n";
}

sub print_stats_ambN
{
    #description_ prints accuracy per ambiguity level
    #param1 _ statistics hash reference

    my $stats = shift;

    my $ambh = $stats->{ambN};

    my $nsamples;
    my $nhits;
    print "* ================= ACCURACY PER LEVEL OF AMBIGUITY =======================================\n";
    print "#CLASSES = ", scalar(keys %{$ambh}), "\n";
    print "* =========================================================================================\n";
    printf STDOUT "%10s %16s %16s  %10s  %10s\n", "LEVEL", "HITS", "TRIALS", "ACCURACY", "MFT";
    print "* -----------------------------------------------------------------------------------------\n";
    foreach my $k (sort{$a <=> $b} keys %{$ambh}) {
        $nhits += $ambh->{$k}->[1];
        $nsamples += $ambh->{$k}->[0];
        printf STDOUT "%10d %16d %16d %10.4f%% %10.4f%%\n", $k, $ambh->{$k}[1], $ambh->{$k}[0], ($ambh->{$k}[0] > 0)? ($ambh->{$k}[1] / $ambh->{$k}[0]) * 100 : 0, ($ambh->{$k}[0] > 0)? ($ambh->{$k}[2] / $ambh->{$k}[0]) * 100 : 0;
    }
}

sub print_stats_ambK
{
    #description_ prints accuracy per ambiguity kind
    #param1 _ statistics hash reference

    my $stats = shift;

    my $ambh = $stats->{ambK};

    my $nsamples;
    my $nhits;
    print "* ================= ACCURACY PER CLASS OF AMBIGUITY =======================================\n";
    print "#CLASSES = ", scalar(keys %{$ambh}), "\n";
    print "* =========================================================================================\n";
    printf STDOUT "%30s %16s %16s  %10s  %10s\n", "CLASS", "HITS", "TRIALS", "ACCURACY", "MFT";
    print "* -----------------------------------------------------------------------------------------\n";
    foreach my $k (sort keys %{$ambh}) {
        $nhits += $ambh->{$k}->[1];
        $nsamples += $ambh->{$k}->[0];
        printf STDOUT "%-30s %16d %16d %10.4f%% %10.4f%%\n", $k, $ambh->{$k}[1], $ambh->{$k}[0], ($ambh->{$k}[0] > 0)? ($ambh->{$k}[1] / $ambh->{$k}[0]) * 100 : 0, ($ambh->{$k}[0] > 0)? ($ambh->{$k}[2] / $ambh->{$k}[0]) * 100 : 0;
    }
}

sub print_stats_ambP
{
    #description_ prints accuracy per part-of-speech
    #param1 _ statistics hash reference

    my $stats = shift;

    my $ambh = $stats->{ambP};

    my $nsamples;
    my $nhits;
    print "* ================= ACCURACY PER PART-OF-SPEECH ===========================================\n";
    printf STDOUT "%10s %16s %16s  %10s  %10s\n", "POS", "HITS", "TRIALS", "ACCURACY", "MFT";
    print "* -----------------------------------------------------------------------------------------\n";
    foreach my $k (sort keys %{$ambh}) {
        $nhits += $ambh->{$k}->[1];
        $nsamples += $ambh->{$k}->[0];
        printf STDOUT "%10s %16d %16d %10.4f%% %10.4f%%\n", $k, $ambh->{$k}[1], $ambh->{$k}[0], ($ambh->{$k}[0] > 0)? ($ambh->{$k}[1] / $ambh->{$k}[0]) * 100 : 0, ($ambh->{$k}[0] > 0)? ($ambh->{$k}[2] / $ambh->{$k}[0]) * 100 : 0;
    }
}

sub print_results
{
    #description_ prints results
    #param1 _ statistics hash reference

    my $stats = shift;

    my ($acckn, $accamb, $accunk, $accuracy) = STATS::get_accuracies($stats);
    print "$acckn\t$accamb\t$accunk\t$accuracy\n";
}

sub get_accuracies
{
    #description_ prints results
    #param1  _ statistics hash reference
    #@return _ (accuracy for known words, accuracy for ambiguous known words,
    #           accuracy for unknown words, overall accuracy)

    my $stats = shift;

    my $nsamples = $stats->{nsamples};
    my $ambiguous = $stats->{namb};
    my $unknown = $stats->{nunk};
    my $nhits = $stats->{nhits};
    my $nambhits = $stats->{nambhits};
    my $nunkhits = $stats->{nunkhits};

    my $acckn = sprintf("%.4f", COMMON::compute_accuracy($nhits - $nunkhits, $nsamples - $unknown) * 100) + 0;
    my $accamb = sprintf("%.4f", COMMON::compute_accuracy($nambhits, $ambiguous) * 100) + 0;
    my $accunk = sprintf("%.4f", COMMON::compute_accuracy($nunkhits, $unknown) * 100) + 0;
    my $accuracy = sprintf("%.4f", COMMON::compute_accuracy($nhits, $nsamples) * 100) + 0;

    return ($acckn, $accamb, $accunk, $accuracy);
}

1;

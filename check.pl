#!/usr/bin/perl -w

use strict;

sub calculateOutput
{
    my ($input, $weights1, $weights2) = @_;

    my $z1 = $input->[0] * $weights1->[0] + $input->[1] * $weights1->[1] + $weights1->[2];
    my $z2 = $input->[0] * $weights2->[0] + $input->[1] * $weights2->[1] + $weights2->[2];

    my $a1 = exp($z1)/(exp($z1) + exp($z2));
    my $a2 = exp($z2)/(exp($z1) + exp($z2));

    my @result = ($a1, $a2);
    return \@result;
}

sub calculateLoss
{
    my ($output, $expected) = @_;

    my $sum = 0.0;

    for(my $i = 0; $i < scalar(@$output); $i++)
    {
        my $expect = $expected->[$i];
        my $actual = $output->[$i];

        $sum += -$expect * log($actual);
    }

    return $sum;
}

sub main
{
    my @weights1 = qw(-0.56208    -0.90591  0.86939);
    my @weights2 = qw(0.35773    +0.35859  -0.23300);
    my @input = qw(0.131538 0.45865);
    my @expected = qw(0 1);

    my $loss = calculateLoss(calculateOutput(\@input, \@weights1, \@weights2), \@expected);

    my $inc = 0.0000001;

    $weights1[2] += $inc;
    my $loss1 = calculateLoss(calculateOutput(\@input, \@weights1, \@weights2), \@expected);
    $weights1[2] -= $inc;

    $weights2[2] += $inc;
    my $loss2 = calculateLoss(calculateOutput(\@input, \@weights1, \@weights2), \@expected);
    $weights2[2] -= $inc;

    my $error1 = ($loss1 - $loss)/$inc;
    my $error2 = ($loss2 - $loss)/$inc;

    print "$error1 $error2\n";
    print "$loss\n";

}

main();


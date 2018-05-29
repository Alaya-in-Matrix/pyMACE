#!/usr/bin/perl
use strict;
use warnings;
use 5.010;

my @params;
my $prob = 5;
$prob    = 3 * ($prob - 1) + 1;

open my $param_f, "<", "./param" or die "Can't open param:$!\n";
while(my $line = <$param_f>)
{
    chomp($line);
    if($line =~ /^\.param.*\=\s*(.*)/)
    {
        push @params, $1;
    }
    else
    {
        say "Invalid line in param:$line";
    }
}
close $param_f;
my $dim = scalar @params;


open my $cec_param, ">", "./cec2014/param" or die "Can't create cec2014/param:$!\n";
say $cec_param $_ for(@params);
close $cec_param;

run_cmd("cd cec2014 && cec2014expensive $prob  > result.po");
open my $fh, "<", "./cec2014/result.po";
chomp(my $fom = <$fh>);
close $fh;

open my $ofh, ">", "result.po";
say $ofh $fom;
close $ofh;
# run_cmd("cp ./cec2014/result.po ./");

sub run_cmd
{
    my $cmd = shift;
    my $ret = system($cmd);
    if($ret != 0)
    {
        die "Fail to run cmd: $cmd\n";
    }
}

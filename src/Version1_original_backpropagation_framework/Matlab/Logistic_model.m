function [] = Logistic_model()
toll_fre=0.2;
trav_fre=0.25;

toll_art=0;
trav_art=0.5;
VOT=0.42;
A=exp(-trav_art*(VOT)-toll_art)/(exp(-trav_art*(VOT)-toll_art)+exp(-trav_fre*(VOT)-toll_fre))
end



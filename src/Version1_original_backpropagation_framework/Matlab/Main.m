unction [TOTAL_ERROR] = Main()
%% Basic setting for sampling
nb_train_sample=1; % total number of samples used for training
nb_test_sample=1;
nb_zone_sample=1; % How many samples of household survey;
nb_od_sample=1; % How many samples of mobile phone data;
nb_sensor_sample=1; % How mand samples of sensor data;
if nb_train_sample>1
 QQ=0; % do not generate sample randomly 
elseif nb_train_sample==1
 QQ=1;
end
%% Parameters setting
Iter_training=300;
Iter_testing=0;

% Weights  of different data sources
Lambda_household=1/3;
Lambda_mobile=1/3;
Lambda_sensor=1/3;

%% Step size of gradient descend 1
 STEP1=1e-1;
 STEP2=1e-1;
 STEP3=1e+6;
%% Step size of gradient descend 2
%  STEP1=1e-2;
%  STEP2=1e-2;
%  STEP3=1e+4;

%% Input_data (The first sample)
fprintf('\n\nData preparation \n\n')
zone_sample_id=1;
od_sample_id=1;
link_sample_id=1;
[g_ozone_vector,g_od_vector,g_path_vector,g_link_vector] ...
            = TNode_Input(zone_sample_id,od_sample_id,link_sample_id);
[sample_ozone,sample_od,sample_link] ...
            = TSample_Input(nb_zone_sample,nb_od_sample,nb_sensor_sample,nb_train_sample, QQ);

        
%% ============ Parameter setting=======================
[~,nb_ozone]=size(g_ozone_vector);
[~,nb_od]=size(g_od_vector);
[~,nb_path]=size(g_path_vector);
[~,nb_link]=size(g_link_vector);
%% ============Initialization================
%max_epoch=3;%最大训练次数
MAPE=[];
TOTAL_ERROR=[];
SURVEY_ERROR=[];
SENSOR_ERROR=[];
CELL_ERROR=[];
Estimated_generation=[];
Estimated_OD_flow=[];
Estimated_gamma=[];
Estimated_theta=[];
Estimated_path_proportion=[];
Estimated_path_flow=[];
Estimated_link_flow=[];
nb_sample=nb_train_sample;
iter =1;
%% Training
%for epoch=1:max_epoch
while iter<=Iter_training+Iter_testing
     if iter>Iter_training
           [sample_ozone,sample_od,sample_link] ...
            = TSample_Input(nb_zone_sample,nb_od_sample,nb_sensor_sample,nb_test_sample,0);
            STEP1=STEP1*0;
            STEP2=STEP2*0;
            STEP3=STEP3*0;
            nb_sample=nb_test_sample;
    end
    for nb=1:nb_sample
        [g_ozone_vector,g_od_vector,g_link_vector]...
                =Sampling(g_ozone_vector,g_od_vector,g_link_vector,nb,sample_ozone,sample_od,sample_link);  
    %BPR function 
        
    %% ============== Forward Pass==============
    fprintf('\n\nForward passing starting....iter*epoch=%d \n\n',iter)
            % forward passing
            [g_od_vector,g_path_vector,g_link_vector]...
                =Forward_passing(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector);
%             for p=1:nb_path
%                 g_path_vector(p)= Update_path_rtt(g_path_vector(p),g_link_vector);
%             end    
           [total_error,survey_error,cell_error,sensor_error,total_mape]...
               =SE_function(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector,sample_ozone,sample_od,sample_link,Lambda_household, Lambda_mobile,Lambda_sensor);
           %%　save output
           Output={[g_ozone_vector.estimated_generation],[g_od_vector.estimated_demand],[g_od_vector.od_estimated_gamma]...
             ,[g_od_vector.estimated_theta],[g_path_vector.path_estimated_proportion],[g_path_vector.path_estimated_flow],[g_link_vector.estimated_flow]};
            Estimated_generation=[Estimated_generation;Output{1}];
            Estimated_OD_flow=[Estimated_OD_flow;Output{2}];
            Estimated_gamma=[Estimated_gamma;Output{3}];
            Estimated_theta=[Estimated_theta;Output{4}];
            Estimated_path_proportion=[Estimated_path_proportion;Output{5}];
            Estimated_path_flow=[Estimated_path_flow;Output{6}];
            Estimated_link_flow=[Estimated_link_flow;Output{7}];
           
           %% ============== Back propagation==============
   
    fprintf('\n\nBack propagation starting....iter*epoch=%d\n\n',iter)
            % Calculate preliminary derivatives
            [g_ozone_vector,g_od_vector,g_path_vector,g_link_vector]...
                =New_Preliminary_derivatives(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector);        
            % Back propagation
            [g_ozone_vector,g_od_vector,g_path_vector,g_link_vector]...
                =New_Back_propagation(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector,Lambda_sensor);


    %% ============== Update parameters==============
            % update the theta
            for od=1:nb_od
                temp_theta=g_od_vector(od).estimated_theta;
                gradient1=g_od_vector(od).link_theta_error_signal;
                g_od_vector(od).estimated_theta=min(inf,max(0.05,g_od_vector(od).estimated_theta-STEP1*gradient1));% 时间价值0.1最小inf元最大）
                if temp_theta<=g_od_vector(od).estimated_theta
                    fprintf('Update theta of od %d, increases %d \n',g_od_vector(od).od_id,-STEP1*gradient1)
                else
                    fprintf('Update theta of od %d, decreases %d \n',g_od_vector(od).od_id,-STEP1*gradient1)
                end
            end
            % update the gamma
            for od=1:nb_od
                temp_gamma=g_od_vector(od).od_estimated_gamma;
                gradient2=g_od_vector(od).link_gamma_error_signal...
                +Lambda_mobile*(g_od_vector(od).od_estimated_gamma/g_od_vector(od).od_target_gamma-1)*(1/g_od_vector(od).od_target_gamma);
                g_od_vector(od).od_estimated_gamma=min(1,max(0,g_od_vector(od).od_estimated_gamma-STEP2*gradient2));% gamma [0,1]
                if temp_gamma<=g_od_vector(od).od_estimated_gamma
                    fprintf('Update gamma of od %d, increases %d \n',g_od_vector(od).od_id,-STEP2*gradient2)
                else
                    fprintf('Update gamma of od %d, decreases %d \n',g_od_vector(od).od_id,-STEP2*gradient2)
                end
            end
           % update the alpha
            for o=1:nb_ozone
                temp_alpha=g_ozone_vector(o).estimated_generation;
                gradient3=g_ozone_vector(o).link_ozone_error_signal...
                    +Lambda_household*(g_ozone_vector(o).estimated_generation/g_ozone_vector(o).target_generation-1)*(1/g_ozone_vector(o).target_generation);
                   %+(g_ozone_vector(o).estimated_generation-g_ozone_vector(o).target_generation);

                g_ozone_vector(o).estimated_generation=min(1e+10,max(1,g_ozone_vector(o).estimated_generation-STEP3*gradient3));
                if temp_alpha<=g_ozone_vector(o).estimated_generation
                    fprintf('Update alpha of zone %d, increases %d \n',g_ozone_vector(o).ozone_id,-STEP3*gradient3)
                else
                    fprintf('Update alpha of zone %d, decreases %d \n',g_ozone_vector(o).ozone_id,-STEP3*gradient3)
                end
            end
           fprintf('Update the travel time of paths\n')
            TOTAL_ERROR=[TOTAL_ERROR,total_error];
            SURVEY_ERROR=[SURVEY_ERROR,survey_error];
            SENSOR_ERROR=[SENSOR_ERROR,sensor_error];
            CELL_ERROR=[CELL_ERROR,cell_error];
            MAPE=[MAPE,total_mape];
            iter=iter+1;
%             if iter >(Iter_training)/2
%             STEP1=STEP1*0.95;
%             STEP2=STEP2*0.95;
%             STEP3=STEP3*0.95;
%             elseif  iter<=(Iter_training)/2
            STEP1=STEP1;
            STEP2=STEP2;
            STEP3=STEP3;
%            end
            plot(SENSOR_ERROR,'r');
            hold on
            plot(TOTAL_ERROR,'k');
            plot(SURVEY_ERROR,'b');
            plot(CELL_ERROR,'g');  
            drawnow

    end
end

%% output
[total_disc,survey_disc,ozone_ef,ozone_tf,cell_disc,od_ef,~,sensor_disc,link_ef,link_tf]=Discrepancy_function(g_ozone_vector,g_od_vector,g_path_vector,g_link_vector);
 GEN_ERROR=sum(TOTAL_ERROR(Iter_training:Iter_training+Iter_testing))/(Iter_testing+1);
 TRA_ERROR= [TOTAL_ERROR(Iter_training),SURVEY_ERROR(Iter_training),CELL_ERROR(Iter_training),SENSOR_ERROR(Iter_training)]; 
 save('small_case(0.33_0.33_0.33).mat')
end



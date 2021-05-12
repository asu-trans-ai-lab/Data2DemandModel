function [sample_ozone,sample_od,sample_link] = TSample_Input(nb_zone_sample,nb_od_sample,nb_sensor_sample,nb_total_sample,QQ)
%% parameters setting
%% connecting data sources
conn=database('ODME_DATA','','');
setdbprefs('DataReturnFormat','structure');
%% construct zones class
CursZone=exec(conn,'select all ID,sample_id,ozone_id,node_id,population,trip_rate from input_ozone');
CursZone=fetch(CursZone);
[nb_sample,~]=size(CursZone.Data.ID);
sample_ozone={};
iter=1;
s=0;
for ss=1:nb_total_sample
    while s==0
    s= ceil(rand*nb_zone_sample);
    end
    g_ozone_vector={};
    for i=1:nb_sample
    if CursZone.Data.sample_id(i)==s
        ozone=COZone; % construct ozone from class COZone
        ozone.ozone_id=CursZone.Data.ozone_id(i);
        ozone.node_id=CursZone.Data.node_id(i);
        ozone.population=CursZone.Data.population(i); 
        ozone.trip_rate=CursZone.Data.trip_rate(i); 
        if QQ==1
            KK=1; % do not generate sample randomly 
        else
            KK=(rand*(1.5-0.5)+0.5); % generate samples based on uniform distribution
        end
        ozone.target_generation=ozone.trip_rate*ozone.population*KK;
        g_ozone_vector=[g_ozone_vector,ozone];
    end
    end
    sample_ozone{iter}=g_ozone_vector; 
    iter=iter+1;
end
fprintf('number of samples of ozones = %d\n',nb_total_sample);


%% construct ODs class
CursOD=exec(conn,'select all ID,sample_id,od_id,from_zone_id,to_zone_id,OD_split,OD_demand from input_od');
CursOD=fetch(CursOD);
[nb_sample,~]=size(CursOD.Data.ID);
sample_od={};
iter=1;
s=0;
for ss=1:nb_total_sample
    while s==0
        s= ceil(rand*nb_od_sample);
    end
        g_od_vector={};
        for i=1:nb_sample
            if CursOD.Data.sample_id(i)==s
            od=COD;% construct object from class COD
            od.od_id=CursOD.Data.od_id(i);
            od.from_zone_id=CursOD.Data.from_zone_id(i);
            od.to_zone_id=CursOD.Data.to_zone_id(i);
            od.target_demand=CursOD.Data.OD_demand(i);
            od.od_target_gamma=CursOD.Data.OD_split(i);
            g_od_vector=[g_od_vector,od];
            end
        end
        sample_od{iter}=g_od_vector;
        iter=iter+1;
end

fprintf('number of samples of ods = %d\n',nb_total_sample);

%% construct sensors class
CursSensor=exec(conn,'select all ID,sample_id,sensor_name,sensor_count from input_sensor');
CursSensor=fetch(CursSensor);
[nb_sample,~]=size(CursSensor.Data.ID);
sample_link={};
iter=1;
s=0;
for ss=1:nb_total_sample
    while s==0
        s= ceil(rand*nb_sensor_sample);
    end
    g_sensor_vector={};  
    nb_sensor=0;
    for i=1:nb_sample
        if s==CursSensor.Data.sample_id(i)
            sensor=CSensor;
            sensor.sensor_name=CursSensor.Data.sensor_name(i);
            if QQ==1
                KK=1; % do not generate sample randomly 
            else
                KK=(rand*(1.5-0.5)+0.5); % generate samples based on uniform distribution
            end
            sensor.sensor_count=CursSensor.Data.sensor_count(i)*KK;
            nb_sensor=nb_sensor+1;
            g_sensor_vector=[g_sensor_vector,sensor];
        end
    end
    %% construct links class
    CursLink=exec(conn,'select all link_id,from_node_id,to_node_id,length,speed_limit,toll,lane_cap,BPR_alpha,BPR_belta,count_sensor_id from input_link');
    CursLink=fetch(CursLink);
    [nb_link,~]=size(CursLink.Data.link_id);
    g_link_vector={};
    for i=1:nb_link
        link=CLink;
        link.link_id=CursLink.Data.link_id(i);
        link.from_node_id=CursLink.Data.from_node_id(i);
        link.to_node_id=CursLink.Data.to_node_id(i);
        link.count_sensor_name=CursLink.Data.count_sensor_id(i);   
        for j=1:nb_sensor
           if size(str2num(link.count_sensor_name{1}),1)~=1 % judge wht
               if link.count_sensor_name{1}==g_sensor_vector(j).sensor_name{1}% get the string
                  link.target_flow=g_sensor_vector(j).sensor_count;
               end
           end
        end      
        g_link_vector=[g_link_vector,link];  
    end
    sample_link{iter}=g_link_vector;
    iter=iter+1;
end
fprintf('number of samples of links = %d\n',nb_total_sample)
end


function [g_ozone_vector,g_od_vector,g_path_vector,g_link_vector] = TNode_Input(zone_sample_id,od_sample_id,sensor_sample_id)
conn=database('ODME_DATA','','');
setdbprefs('DataReturnFormat','structure');


%% construct nodes class
CursNode=exec(conn,'select all zone_id,node_id from input_node');
CursNode=fetch(CursNode);
[nb_node,~]=size(CursNode.Data.node_id);
g_node_vector={};
for i=1:nb_node
    node=CNode; % construct node from class CNode
    node.node_id=i;
    node.zone_id=CursNode.Data.zone_id(i);
    g_node_vector=[g_node_vector,node];
end

fprintf('number of nodes = %d\n',nb_node);

%% construct zones class
CursZone=exec(conn,'select all ID,sample_id,ozone_id,node_id,population,trip_rate from input_ozone');
CursZone=fetch(CursZone);
[nb_sample,~]=size(CursZone.Data.ID);
g_ozone_vector={};
nb_ozone=0;
for i=1:nb_sample
    if CursZone.Data.sample_id(i)==zone_sample_id
        ozone=COZone; % construct ozone from class COZone
        ozone.ozone_id=CursZone.Data.ozone_id(i);
        ozone.node_id=CursZone.Data.node_id(i);
        ozone.population=CursZone.Data.population(i); 
        ozone.trip_rate=CursZone.Data.trip_rate(i); 
        ozone=Ref_Gen(ozone,ozone.trip_rate,ozone.population);
        % a method to generate target trip generation and the initial
        % estimated generation volume
        g_ozone_vector=[g_ozone_vector,ozone];
        nb_ozone=nb_ozone+1;
    end
end

fprintf('number of ozones = %d\n',nb_ozone);


%% construct ODs class
CursOD=exec(conn,'select all ID,sample_id,od_id,from_zone_id,to_zone_id,OD_split,OD_demand from input_od');
CursOD=fetch(CursOD);
[nb_sample,~]=size(CursOD.Data.ID);
g_od_vector={};
nb_od=0;
for i=1:nb_sample
    if CursOD.Data.sample_id(i)==od_sample_id
    od=COD;% construct object from class COD
    od.od_id=CursOD.Data.od_id(i);
    od.from_zone_id=CursOD.Data.from_zone_id(i);
    od.to_zone_id=CursOD.Data.to_zone_id(i);
    od.target_demand=CursOD.Data.OD_demand(i);
    od.estimated_theta=1; % assumed theta ==1
    od.od_target_gamma=CursOD.Data.OD_split(i);
    od.od_estimated_gamma=1; %rand; % using rand seed to generate gamma 
    g_od_vector=[g_od_vector,od];
    nb_od=nb_od+1;
    end
end

fprintf('number of ods = %d\n',nb_od);

%% construct sensors class
CursSensor=exec(conn,'select all ID,sample_id,sensor_name,sensor_count from input_sensor');
CursSensor=fetch(CursSensor);
[nb_sample,~]=size(CursSensor.Data.ID);
g_sensor_vector={};
nb_sensor=0;
for i=1:nb_sample
    if sensor_sample_id==CursSensor.Data.sample_id(i)
        sensor=CSensor;
        sensor.sensor_name=CursSensor.Data.sensor_name(i);
        sensor.sensor_count=CursSensor.Data.sensor_count(i);
        nb_sensor=nb_sensor+1;
        g_sensor_vector=[g_sensor_vector,sensor];
    end
end
fprintf('number of sensors = %d\n',nb_sensor); % %d：表示整数；\n表示换行


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
    link.length=CursLink.Data.length(i);
    link.speed_limit=CursLink.Data.speed_limit(i);
    link.lane_cap=CursLink.Data.lane_cap(i);
    link.toll=CursLink.Data.toll(i);
    link.BPR_alpha=CursLink.Data.BPR_alpha(i);
    link.BPR_belta=CursLink.Data.BPR_belta(i);
    link.count_sensor_name=CursLink.Data.count_sensor_id(i);   
    for j=1:nb_sensor
       if size(str2num(link.count_sensor_name{1}),1)~=1 % judge wht
           if link.count_sensor_name{1}==g_sensor_vector(j).sensor_name{1}% get the string
              link.target_flow=g_sensor_vector(j).sensor_count;
              link.estimated_flow=g_sensor_vector(j).sensor_count;
           end
       end
    end      
    link=Calculate_FFTT(link);
    link.real_travel_time=link.free_flow_travel_time;% initialize the real travel time (rtt) 
    g_link_vector=[g_link_vector,link];  
end

fprintf('number of links = %d\n',nb_link)
%% construct paths class
CursPath=exec(conn,'select all ID,from_zone_id,to_zone_id,K,node_sequence from input_path');
CursPath=fetch(CursPath);
[nb_path,~]=size(CursPath.Data.ID);
g_path_vector={};
for i=1:nb_path
    path=CPath(i,nb_link);
    path.path_id=CursPath.Data.ID(i);
    path.from_zone_id=CursPath.Data.from_zone_id(i);
    path.to_zone_id=CursPath.Data.to_zone_id(i);
    path.K=CursPath.Data.K(i);
    path.node_sequence=str2num(CursPath.Data.node_sequence{i});
    path=node2link(path,path.node_sequence);
    path.path_cost=0;
    path.path_travel_time=0;
    path= Update_path_rtt(path,g_link_vector);
    g_path_vector=[g_path_vector,path];
end


fprintf('number of paths = %d\n',nb_path)
end


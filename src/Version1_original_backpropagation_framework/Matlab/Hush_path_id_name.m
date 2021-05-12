function [ output_args ] = Hush_path_id_name( input_args )
PATH_ID={};
PATH_NAME={};
for i=1:nb_path
PATH_ID=[PATH_ID,i];
PATH_NAME=[PATH_NAME,CursPath.Data.node_sequence(i)];
end
map_path_id_name=containers.Map(PATH_ID,PATH_NAME);



end


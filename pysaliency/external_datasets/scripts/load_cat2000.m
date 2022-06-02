load trainSet/allFixData.mat;
ind = 0
ks=keys(allData);
for i=1:size(ks,2);
	k=ks(i);
	tmp=allData(char(k));
	for j=1:size(tmp,1);
		ttmp=cell2mat(tmp(j));
		name = ttmp.name;
		data = ttmp.data;
		filename = sprintf('extracted/fix%d_%d.mat', i, j);
		save(filename, 'name', 'data');
		ind = ind+1;
		disp(sprintf('%d/%d', i,j));
	end;
end

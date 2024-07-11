select m.run_uuid 
	from metrics as m
	JOIN public.runs AS r ON r.run_uuid = m.run_uuid	
	JOIN public.experiments AS exp ON r.experiment_id = exp.experiment_id
	where exp.experiment_id > 0
	AND exp.name NOT LIKE '%output%' 
    AND exp.lifecycle_stage = 'active'
	and ( key like 'Perf')
	order by value desc
	limit 5
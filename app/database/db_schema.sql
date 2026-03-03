create table if not exists students (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  reg_no text unique not null,
  created_at timestamp default now()
);

create table if not exists student_faces (
  id uuid primary key default gen_random_uuid(),
  student_id uuid references students(id) on delete cascade,
  embedding jsonb not null,
  created_at timestamp default now()
);

create table if not exists attendance (
  id uuid primary key default gen_random_uuid(),
  student_id uuid references students(id) on delete cascade,
  date date not null,
  time timestamp default now(),
  status text default 'present'
);

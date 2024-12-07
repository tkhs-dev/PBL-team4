create table if not exists assignment(
    id serial primary key,
    assigned_at timestamp not null,
    client varchar(255) not null,
    deadline timestamp not null,
    status enum('processing', 'completed', 'timeup', 'error') not null,
    status_changed_at timestamp not null,
    task uuid not null,
)
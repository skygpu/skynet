create db in postgres:

```sql
CREATE USER skynet WITH PASSWORD 'password';
CREATE DATABASE skynet_art_bot;
GRANT ALL PRIVILEGES ON DATABASE skynet_art_bot TO skynet;

CREATE SCHEMA IF NOT EXISTS skynet;

CREATE TABLE IF NOT EXISTS skynet.user(
   id SERIAL PRIMARY KEY NOT NULL,
   tg_id INT,
   wp_id VARCHAR(128),
   mx_id VARCHAR(128),
   ig_id VARCHAR(128),
   generated INT NOT NULL,
   joined DATE NOT NULL,
   last_prompt TEXT,
   role VARCHAR(128) NOT NULL
);
ALTER TABLE skynet.user
    ADD CONSTRAINT tg_unique
    UNIQUE (tg_id);
ALTER TABLE skynet.user
    ADD CONSTRAINT wp_unique
    UNIQUE (wp_id);
ALTER TABLE skynet.user
    ADD CONSTRAINT mx_unique
    UNIQUE (mx_id);
ALTER TABLE skynet.user
    ADD CONSTRAINT ig_unique
    UNIQUE (ig_id);

CREATE TABLE IF NOT EXISTS skynet.user_config(
    id SERIAL NOT NULL,
    algo VARCHAR(128) NOT NULL,
    step INT NOT NULL,
    width INT NOT NULL,
    height INT NOT NULL,
    seed INT,
    guidance INT NOT NULL,
    upscaler VARCHAR(128)
);
ALTER TABLE skynet.user_config
    ADD FOREIGN KEY(id)
    REFERENCES skynet.user(id);
```

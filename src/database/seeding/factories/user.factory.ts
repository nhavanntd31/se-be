import { setSeederFactory } from 'typeorm-extension';
import { User } from 'src/database/entities';

export default setSeederFactory(User, (faker) => {
  const user = new User();

  user.name = String(faker.datatype.number(10000));
  user.email = faker.internet.email();

  return user;
});

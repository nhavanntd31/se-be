import { Logger } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { InjectRepository } from '@nestjs/typeorm';
import {
  OnGatewayConnection,
  OnGatewayDisconnect,
  OnGatewayInit,
  WebSocketGateway,
  WebSocketServer,
} from '@nestjs/websockets';
import { Socket, Server } from 'socket.io';
import { User } from 'src/database/entities';
import { Repository } from 'typeorm';

@WebSocketGateway({
  cors: true,
})
export class SocketGateway
  implements OnGatewayInit, OnGatewayConnection, OnGatewayDisconnect
{
  private readonly logger = new Logger(SocketGateway.name);

  constructor(
    @InjectRepository(User)
    private readonly userRepository: Repository<User>,
    private readonly jwtService: JwtService,
  ) {}

  @WebSocketServer()
  server: Server;

  afterInit() {
    this.logger.log('Initialize WebSocket');
  }

  handleDisconnect(client: Socket) {
    const userId = client.handshake.query?.userId?.toString();
    if (userId) {
      this.logger.log(`Client disconnected: ${userId}`);
    } else {
      this.logger.log(`Client disconnected: ${client.id}`);
    }
  }

  async handleConnection(client: Socket, ...args: any[]) {}
}

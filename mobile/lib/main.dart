// main.dart - Enhanced Flutter App με Portfolio Tracking & Technical Analysis
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';

// Configuration
class AppConfig {
  static String get apiUrl {
    const bool isProduction = bool.fromEnvironment('dart.vm.product');
    if (isProduction) {
      return 'https://api.stockalertpro.com';  // Your production URL
    }
    return 'http://localhost:8000';  // Development
  }
  
  static const String appName = 'StockAlert Pro';
  static const String version = '2.0.0';
}

// Theme Configuration
class AppTheme {
  static const Color primaryGreen = Color(0xFF00C853);
  static const Color primaryRed = Color(0xFFD50000);
  static const Color primaryBlue = Color(0xFF2196F3);
  static const Color bgDark = Color(0xFF0A0E21);
  static const Color cardDark = Color(0xFF1C2333);
  
  static ThemeData get darkTheme => ThemeData(
    primarySwatch: Colors.blue,
    scaffoldBackgroundColor: bgDark,
    brightness: Brightness.dark,
    cardColor: cardDark,
    appBarTheme: AppBarTheme(
      backgroundColor: cardDark,
      elevation: 0,
    ),
  );
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(StockAlertProApp());
}

class StockAlertProApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(create: (_) => PortfolioProvider()),
        ChangeNotifierProvider(create: (_) => WatchlistProvider()),
      ],
      child: MaterialApp(
        title: AppConfig.appName,
        theme: AppTheme.darkTheme,
        home: AuthWrapper(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}

// Auth Provider
class AuthProvider extends ChangeNotifier {
  User? _user;
  String? _authToken;
  
  User? get user => _user;
  String? get authToken => _authToken;
  bool get isAuthenticated => _user != null;
  
  AuthProvider() {
    FirebaseAuth.instance.authStateChanges().listen((User? user) async {
      _user = user;
      if (user != null) {
        _authToken = await user.getIdToken();
      } else {
        _authToken = null;
      }
      notifyListeners();
    });
  }
  
  Future<void> signIn(String email, String password) async {
    try {
      await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
    } catch (e) {
      throw Exception('Sign in failed: $e');
    }
  }
  
  Future<void> signUp(String email, String password) async {
    try {
      await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );
    } catch (e) {
      throw Exception('Sign up failed: $e');
    }
  }
  
  Future<void> signOut() async {
    await FirebaseAuth.instance.signOut();
  }
}

// Portfolio Provider
class PortfolioProvider extends ChangeNotifier {
  Map<String, dynamic> _portfolio = {};
  List<Position> _positions = [];
  List<Transaction> _transactions = [];
  bool _isLoading = false;
  
  Map<String, dynamic> get portfolio => _portfolio;
  List<Position> get positions => _positions;
  List<Transaction> get transactions => _transactions;
  bool get isLoading => _isLoading;
  
  double get totalValue => _portfolio['total_value'] ?? 0.0;
  double get totalGainLoss => _portfolio['total_gain_loss'] ?? 0.0;
  double get totalGainLossPercent => _portfolio['total_gain_loss_percent'] ?? 0.0;
  
  Future<void> loadPortfolio(String authToken) async {
    _isLoading = true;
    notifyListeners();
    
    try {
      final response = await http.get(
        Uri.parse('${AppConfig.apiUrl}/api/portfolio'),
        headers: {
          'Authorization': 'Bearer $authToken',
          'Content-Type': 'application/json',
        },
      );
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _portfolio = data;
        _positions = (data['positions'] as List)
            .map((p) => Position.fromJson(p))
            .toList();
      }
    } catch (e) {
      print('Error loading portfolio: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
  
  Future<void> addTransaction(String authToken, Transaction transaction) async {
    try {
      final response = await http.post(
        Uri.parse('${AppConfig.apiUrl}/api/portfolio/transaction'),
        headers: {
          'Authorization': 'Bearer $authToken',
          'Content-Type': 'application/json',
        },
        body: json.encode(transaction.toJson()),
      );
      
      if (response.statusCode == 200) {
        await loadPortfolio(authToken);
      }
    } catch (e) {
      print('Error adding transaction: $e');
      throw e;
    }
  }
}

// Watchlist Provider
class WatchlistProvider extends ChangeNotifier {
  List<String> _watchlist = [];
  Map<String, StockData> _stockData = {};
  
  List<String> get watchlist => _watchlist;
  Map<String, StockData> get stockData => _stockData;
  
  Future<void> loadWatchlist() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    _watchlist = prefs.getStringList('watchlist') ?? [];
    notifyListeners();
    
    // Load stock data for watchlist
    for (String symbol in _watchlist) {
      await loadStockData(symbol);
    }
  }
  
  Future<void> addToWatchlist(String symbol) async {
    if (!_watchlist.contains(symbol)) {
      _watchlist.add(symbol);
      SharedPreferences prefs = await SharedPreferences.getInstance();
      await prefs.setStringList('watchlist', _watchlist);
      await loadStockData(symbol);
      notifyListeners();
    }
  }
  
  Future<void> removeFromWatchlist(String symbol) async {
    _watchlist.remove(symbol);
    _stockData.remove(symbol);
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('watchlist', _watchlist);
    notifyListeners();
  }
  
  Future<void> loadStockData(String symbol) async {
    try {
      final response = await http.get(
        Uri.parse('${AppConfig.apiUrl}/api/stock/$symbol/price'),
      );
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _stockData[symbol] = StockData.fromJson(data);
        notifyListeners();
      }
    } catch (e) {
      print('Error loading stock data for $symbol: $e');
    }
  }
}

// Models
class Position {
  final String symbol;
  final double shares;
  final double averagePrice;
  final double currentPrice;
  final double totalValue;
  final double gainLoss;
  final double gainLossPercent;
  
  Position({
    required this.symbol,
    required this.shares,
    required this.averagePrice,
    required this.currentPrice,
    required this.totalValue,
    required this.gainLoss,
    required this.gainLossPercent,
  });
  
  factory Position.fromJson(Map<String, dynamic> json) {
    return Position(
      symbol: json['symbol'],
      shares: json['shares'].toDouble(),
      averagePrice: json['average_price'].toDouble(),
      currentPrice: json['current_price']?.toDouble() ?? 0,
      totalValue: json['total_value']?.toDouble() ?? 0,
      gainLoss: json['gain_loss']?.toDouble() ?? 0,
      gainLossPercent: json['gain_loss_percent']?.toDouble() ?? 0,
    );
  }
}

class Transaction {
  final String symbol;
  final String type;
  final double shares;
  final double price;
  final DateTime timestamp;
  final double commission;
  
  Transaction({
    required this.symbol,
    required this.type,
    required this.shares,
    required this.price,
    required this.timestamp,
    this.commission = 0,
  });
  
  Map<String, dynamic> toJson() {
    return {
      'symbol': symbol,
      'type': type,
      'shares': shares,
      'price': price,
      'timestamp': timestamp.toIso8601String(),
      'commission': commission,
    };
  }
  
  factory Transaction.fromJson(Map<String, dynamic> json) {
    return Transaction(
      symbol: json['symbol'],
      type: json['type'],
      shares: json['shares'].toDouble(),
      price: json['price'].toDouble(),
      timestamp: DateTime.parse(json['timestamp']),
      commission: json['commission']?.toDouble() ?? 0,
    );
  }
}

class StockData {
  final String symbol;
  final double price;
  final double dayChange;
  final double dayChangePercent;
  final int volume;
  final double marketCap;
  
  StockData({
    required this.symbol,
    required this.price,
    required this.dayChange,
    required this.dayChangePercent,
    required this.volume,
    required this.marketCap,
  });
  
  factory StockData.fromJson(Map<String, dynamic> json) {
    return StockData(
      symbol: json['symbol'],
      price: json['price']?.toDouble() ?? 0,
      dayChange: json['dayChange']?.toDouble() ?? 0,
      dayChangePercent: json['dayChangePercent']?.toDouble() ?? 0,
      volume: json['volume'] ?? 0,
      marketCap: json['marketCap']?.toDouble() ?? 0,
    );
  }
}

class TechnicalIndicators {
  final double rsi;
  final Map<String, double> macd;
  final Map<String, double> bollingerBands;
  final double sma20;
  final double sma50;
  final double sma200;
  
  TechnicalIndicators({
    required this.rsi,
    required this.macd,
    required this.bollingerBands,
    required this.sma20,
    required this.sma50,
    required this.sma200,
  });
  
  factory TechnicalIndicators.fromJson(Map<String, dynamic> json) {
    return TechnicalIndicators(
      rsi: json['rsi']?.toDouble() ?? 50,
      macd: Map<String, double>.from(json['macd'] ?? {}),
      bollingerBands: Map<String, double>.from(json['bollinger_bands'] ?? {}),
      sma20: json['sma_20']?.toDouble() ?? 0,
      sma50: json['sma_50']?.toDouble() ?? 0,
      sma200: json['sma_200']?.toDouble() ?? 0,
    );
  }
}

// Auth Wrapper
class AuthWrapper extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final authProvider = Provider.of<AuthProvider>(context);
    
    if (authProvider.isAuthenticated) {
      return MainScreen();
    } else {
      return LoginScreen();
    }
  }
}

// Login Screen
class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isSignUp = false;
  bool _isLoading = false;
  
  void _authenticate() async {
    if (_emailController.text.isEmpty || _passwordController.text.isEmpty) {
      _showError('Please fill all fields');
      return;
    }
    
    setState(() => _isLoading = true);
    
    try {
      final authProvider = Provider.of<AuthProvider>(context, listen: false);
      
      if (_isSignUp) {
        await authProvider.signUp(
          _emailController.text.trim(),
          _passwordController.text,
        );
      } else {
        await authProvider.signIn(
          _emailController.text.trim(),
          _passwordController.text,
        );
      }
    } catch (e) {
      _showError(e.toString());
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }
  
  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
      ),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              AppTheme.bgDark,
              Color(0xFF151E3D),
            ],
          ),
        ),
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: EdgeInsets.all(24),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Logo
                  Icon(
                    Icons.trending_up,
                    size: 80,
                    color: AppTheme.primaryBlue,
                  ),
                  SizedBox(height: 16),
                  Text(
                    'StockAlert Pro',
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'AI-Powered Stock Trading Signals',
                    style: TextStyle(
                      color: Colors.grey[400],
                    ),
                  ),
                  SizedBox(height: 48),
                  
                  // Email Field
                  TextField(
                    controller: _emailController,
                    decoration: InputDecoration(
                      hintText: 'Email',
                      prefixIcon: Icon(Icons.email),
                      filled: true,
                      fillColor: AppTheme.cardDark,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide.none,
                      ),
                    ),
                    keyboardType: TextInputType.emailAddress,
                  ),
                  SizedBox(height: 16),
                  
                  // Password Field
                  TextField(
                    controller: _passwordController,
                    decoration: InputDecoration(
                      hintText: 'Password',
                      prefixIcon: Icon(Icons.lock),
                      filled: true,
                      fillColor: AppTheme.cardDark,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide.none,
                      ),
                    ),
                    obscureText: true,
                  ),
                  SizedBox(height: 24),
                  
                  // Auth Button
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: ElevatedButton(
                      onPressed: _isLoading ? null : _authenticate,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.primaryBlue,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: _isLoading
                          ? CircularProgressIndicator(color: Colors.white)
                          : Text(
                              _isSignUp ? 'Sign Up' : 'Sign In',
                              style: TextStyle(fontSize: 16),
                            ),
                    ),
                  ),
                  SizedBox(height: 16),
                  
                  // Toggle Auth Mode
                  TextButton(
                    onPressed: () {
                      setState(() => _isSignUp = !_isSignUp);
                    },
                    child: Text(
                      _isSignUp
                          ? 'Already have an account? Sign In'
                          : "Don't have an account? Sign Up",
                      style: TextStyle(color: AppTheme.primaryBlue),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// Main Screen with Navigation
class MainScreen extends StatefulWidget {
  @override
  _MainScreenState createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;
  
  final List<Widget> _screens = [
    PortfolioScreen(),
    WatchlistScreen(),
    SearchScreen(),
    TechnicalAnalysisScreen(),
    SettingsScreen(),
  ];
  
  @override
  void initState() {
    super.initState();
    _loadInitialData();
    _setupNotifications();
  }
  
  void _loadInitialData() async {
    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    final portfolioProvider = Provider.of<PortfolioProvider>(context, listen: false);
    final watchlistProvider = Provider.of<WatchlistProvider>(context, listen: false);
    
    if (authProvider.authToken != null) {
      await portfolioProvider.loadPortfolio(authProvider.authToken!);
    }
    await watchlistProvider.loadWatchlist();
  }
  
  void _setupNotifications() async {
    FirebaseMessaging messaging = FirebaseMessaging.instance;
    
    NotificationSettings settings = await messaging.requestPermission(
      alert: true,
      badge: true,
      sound: true,
    );
    
    if (settings.authorizationStatus == AuthorizationStatus.authorized) {
      String? token = await messaging.getToken();
      print('FCM Token: $token');
      
      FirebaseMessaging.onMessage.listen((RemoteMessage message) {
        _showNotification(message);
      });
    }
  }
  
  void _showNotification(RemoteMessage message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(message.notification?.title ?? 'Alert'),
        content: Text(message.notification?.body ?? ''),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('OK'),
          ),
        ],
      ),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) => setState(() => _currentIndex = index),
        type: BottomNavigationBarType.fixed,
        backgroundColor: AppTheme.cardDark,
        selectedItemColor: AppTheme.primaryBlue,
        unselectedItemColor: Colors.grey,
        items: [
          BottomNavigationBarItem(
            icon: Icon(Icons.account_balance_wallet),
            label: 'Portfolio',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.list),
            label: 'Watchlist',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.search),
            label: 'Search',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.analytics),
            label: 'Analysis',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}

// Portfolio Screen
class PortfolioScreen extends StatefulWidget {
  @override
  _PortfolioScreenState createState() => _PortfolioScreenState();
}

class _PortfolioScreenState extends State<PortfolioScreen> {
  @override
  Widget build(BuildContext context) {
    final portfolioProvider = Provider.of<PortfolioProvider>(context);
    
    return Scaffold(
      appBar: AppBar(
        title: Text('Portfolio'),
        actions: [
          IconButton(
            icon: Icon(Icons.add),
            onPressed: () => _showAddTransactionDialog(context),
          ),
          IconButton(
            icon: Icon(Icons.refresh),
            onPressed: () async {
              final authProvider = Provider.of<AuthProvider>(context, listen: false);
              if (authProvider.authToken != null) {
                await portfolioProvider.loadPortfolio(authProvider.authToken!);
              }
            },
          ),
        ],
      ),
      body: portfolioProvider.isLoading
          ? Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Portfolio Summary Card
                  Card(
                    color: AppTheme.cardDark,
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Total Portfolio Value',
                            style: TextStyle(color: Colors.grey),
                          ),
                          SizedBox(height: 8),
                          Text(
                            NumberFormat.currency(symbol: '\$').format(portfolioProvider.totalValue),
                            style: TextStyle(
                              fontSize: 32,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          SizedBox(height: 8),
                          Row(
                            children: [
                              Icon(
                                portfolioProvider.totalGainLoss >= 0
                                    ? Icons.trending_up
                                    : Icons.trending_down,
                                color: portfolioProvider.totalGainLoss >= 0
                                    ? AppTheme.primaryGreen
                                    : AppTheme.primaryRed,
                              ),
                              SizedBox(width: 8),
                              Text(
                                '${portfolioProvider.totalGainLoss >= 0 ? '+' : ''}${NumberFormat.currency(symbol: '\$').format(portfolioProvider.totalGainLoss)}',
                                style: TextStyle(
                                  color: portfolioProvider.totalGainLoss >= 0
                                      ? AppTheme.primaryGreen
                                      : AppTheme.primaryRed,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              SizedBox(width: 8),
                              Text(
                                '(${portfolioProvider.totalGainLossPercent >= 0 ? '+' : ''}${portfolioProvider.totalGainLossPercent.toStringAsFixed(2)}%)',
                                style: TextStyle(
                                  color: portfolioProvider.totalGainLoss >= 0
                                      ? AppTheme.primaryGreen
                                      : AppTheme.primaryRed,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                  
                  SizedBox(height: 16),
                  
                  // Portfolio Chart
                  Card(
                    color: AppTheme.cardDark,
                    child: Container(
                      height: 200,
                      padding: EdgeInsets.all(16),
                      child: _buildPortfolioChart(portfolioProvider.positions),
                    ),
                  ),
                  
                  SizedBox(height: 16),
                  
                  // Positions List
                  Text(
                    'Positions',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 12),
                  
                  if (portfolioProvider.positions.isEmpty)
                    Center(
                      child: Column(
                        children: [
                          Icon(Icons.account_balance_wallet, size: 64, color: Colors.grey),
                          SizedBox(height: 16),
                          Text(
                            'No positions yet',
                            style: TextStyle(color: Colors.grey),
                          ),
                          SizedBox(height: 8),
                          ElevatedButton.icon(
                            onPressed: () => _showAddTransactionDialog(context),
                            icon: Icon(Icons.add),
                            label: Text('Add Transaction'),
                          ),
                        ],
                      ),
                    )
                  else
                    ...portfolioProvider.positions.map((position) => Card(
                      color: AppTheme.cardDark,
                      margin: EdgeInsets.only(bottom: 8),
                      child: ListTile(
                        leading: Container(
                          width: 48,
                          height: 48,
                          decoration: BoxDecoration(
                            color: position.gainLoss >= 0
                                ? AppTheme.primaryGreen.withOpacity(0.2)
                                : AppTheme.primaryRed.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Center(
                            child: Text(
                              position.symbol.substring(0, min(3, position.symbol.length)),
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: position.gainLoss >= 0
                                    ? AppTheme.primaryGreen
                                    : AppTheme.primaryRed,
                              ),
                            ),
                          ),
                        ),
                        title: Text(
                          position.symbol,
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        subtitle: Text(
                          '${position.shares} shares @ \$${position.averagePrice.toStringAsFixed(2)}',
                          style: TextStyle(color: Colors.grey),
                        ),
                        trailing: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              NumberFormat.currency(symbol: '\$').format(position.totalValue),
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
                            Text(
                              '${position.gainLoss >= 0 ? '+' : ''}${position.gainLossPercent.toStringAsFixed(2)}%',
                              style: TextStyle(
                                color: position.gainLoss >= 0
                                    ? AppTheme.primaryGreen
                                    : AppTheme.primaryRed,
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => PositionDetailScreen(position: position),
                            ),
                          );
                        },
                      ),
                    )).toList(),
                ],
              ),
            ),
    );
  }
  
  Widget _buildPortfolioChart(List<Position> positions) {
    if (positions.isEmpty) {
      return Center(
        child: Text('No data to display', style: TextStyle(color: Colors.grey)),
      );
    }
    
    return PieChart(
      PieChartData(
        sections: positions.map((position) {
          final percentage = (position.totalValue / positions.fold(0.0, (sum, p) => sum + p.totalValue)) * 100;
          return PieChartSectionData(
            value: position.totalValue,
            title: '${position.symbol}\n${percentage.toStringAsFixed(1)}%',
            color: _getColorForIndex(positions.indexOf(position)),
            radius: 80,
            titleStyle: TextStyle(fontSize: 10, color: Colors.white),
          );
        }).toList(),
        sectionsSpace: 2,
        centerSpaceRadius: 0,
      ),
    );
  }
  
  Color _getColorForIndex(int index) {
    final colors = [
      AppTheme.primaryBlue,
      AppTheme.primaryGreen,
      Colors.orange,
      Colors.purple,
      Colors.teal,
      Colors.pink,
      Colors.amber,
      Colors.cyan,
    ];
    return colors[index % colors.length];
  }
  
  void _showAddTransactionDialog(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppTheme.cardDark,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => AddTransactionDialog(),
    );
  }
  
  int min(int a, int b) => a < b ? a : b;
}

// Add Transaction Dialog
class AddTransactionDialog extends StatefulWidget {
  @override
  _AddTransactionDialogState createState() => _AddTransactionDialogState();
}

class _AddTransactionDialogState extends State<AddTransactionDialog> {
  final _symbolController = TextEditingController();
  final _sharesController = TextEditingController();
  final _priceController = TextEditingController();
  String _transactionType = 'BUY';
  
  void _submitTransaction() async {
    if (_symbolController.text.isEmpty ||
        _sharesController.text.isEmpty ||
        _priceController.text.isEmpty) {
      return;
    }
    
    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    final portfolioProvider = Provider.of<PortfolioProvider>(context, listen: false);
    
    if (authProvider.authToken == null) return;
    
    final transaction = Transaction(
      symbol: _symbolController.text.toUpperCase(),
      type: _transactionType,
      shares: double.parse(_sharesController.text),
      price: double.parse(_priceController.text),
      timestamp: DateTime.now(),
    );
    
    try {
      await portfolioProvider.addTransaction(authProvider.authToken!, transaction);
      Navigator.pop(context);
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Transaction added successfully'),
          backgroundColor: AppTheme.primaryGreen,
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e'),
          backgroundColor: AppTheme.primaryRed,
        ),
      );
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
      ),
      child: Container(
        padding: EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Add Transaction',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 24),
            
            // Transaction Type
            Row(
              children: [
                Expanded(
                  child: RadioListTile<String>(
                    title: Text('Buy'),
                    value: 'BUY',
                    groupValue: _transactionType,
                    onChanged: (value) => setState(() => _transactionType = value!),
                    activeColor: AppTheme.primaryGreen,
                  ),
                ),
                Expanded(
                  child: RadioListTile<String>(
                    title: Text('Sell'),
                    value: 'SELL',
                    groupValue: _transactionType,
                    onChanged: (value) => setState(() => _transactionType = value!),
                    activeColor: AppTheme.primaryRed,
                  ),
                ),
              ],
            ),
            
            SizedBox(height: 16),
            
            // Symbol
            TextField(
              controller: _symbolController,
              decoration: InputDecoration(
                labelText: 'Symbol',
                hintText: 'AAPL',
                border: OutlineInputBorder(),
              ),
              textCapitalization: TextCapitalization.characters,
            ),
            
            SizedBox(height: 16),
            
            // Shares
            TextField(
              controller: _sharesController,
              decoration: InputDecoration(
                labelText: 'Number of Shares',
                hintText: '100',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.number,
            ),
            
            SizedBox(height: 16),
            
            // Price
            TextField(
              controller: _priceController,
              decoration: InputDecoration(
                labelText: 'Price per Share',
                hintText: '150.00',
                prefixText: '\$ ',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.numberWithOptions(decimal: true),
            ),
            
            SizedBox(height: 24),
            
            // Submit Button
            SizedBox(
              width: double.infinity,
              height: 48,
              child: ElevatedButton(
                onPressed: _submitTransaction,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _transactionType == 'BUY'
                      ? AppTheme.primaryGreen
                      : AppTheme.primaryRed,
                ),
                child: Text(
                  _transactionType == 'BUY' ? 'Buy Shares' : 'Sell Shares',
                  style: TextStyle(fontSize: 16),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// Technical Analysis Screen
class TechnicalAnalysisScreen extends StatefulWidget {
  @override
  _TechnicalAnalysisScreenState createState() => _TechnicalAnalysisScreenState();
}

class _TechnicalAnalysisScreenState extends State<TechnicalAnalysisScreen> {
  String _selectedSymbol = '';
  TechnicalIndicators? _indicators;
  Map<String, dynamic>? _signals;
  Map<String, dynamic>? _backtestResults;
  bool _isLoading = false;
  
  final _symbolController = TextEditingController();
  
  void _loadTechnicalData() async {
    if (_symbolController.text.isEmpty) return;
    
    setState(() {
      _isLoading = true;
      _selectedSymbol = _symbolController.text.toUpperCase();
    });
    
    try {
      // Load technical indicators
      final indicatorsResponse = await http.get(
        Uri.parse('${AppConfig.apiUrl}/api/technical/${_selectedSymbol}'),
      );
      
      if (indicatorsResponse.statusCode == 200) {
        setState(() {
          _indicators = TechnicalIndicators.fromJson(json.decode(indicatorsResponse.body));
        });
      }
      
      // Load signals
      final signalsResponse = await http.get(
        Uri.parse('${AppConfig.apiUrl}/api/technical/${_selectedSymbol}/signals'),
      );
      
      if (signalsResponse.statusCode == 200) {
        setState(() {
          _signals = json.decode(signalsResponse.body);
        });
      }
      
      // Load backtest results
      final backtestResponse = await http.get(
        Uri.parse('${AppConfig.apiUrl}/api/backtest/${_selectedSymbol}/compare'),
      );
      
      if (backtestResponse.statusCode == 200) {
        setState(() {
          _backtestResults = json.decode(backtestResponse.body);
        });
      }
    } catch (e) {
      print('Error loading technical data: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Technical Analysis'),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Symbol Input
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _symbolController,
                    decoration: InputDecoration(
                      hintText: 'Enter symbol (e.g., AAPL)',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      filled: true,
                      fillColor: AppTheme.cardDark,
                    ),
                    textCapitalization: TextCapitalization.characters,
                  ),
                ),
                SizedBox(width: 12),
                ElevatedButton(
                  onPressed: _isLoading ? null : _loadTechnicalData,
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                  ),
                  child: _isLoading
                      ? CircularProgressIndicator(color: Colors.white, strokeWidth: 2)
                      : Text('Analyze'),
                ),
              ],
            ),
            
            if (_selectedSymbol.isNotEmpty) ...[
              SizedBox(height: 24),
              
              // Trading Signal Card
              if (_signals != null) _buildSignalCard(),
              
              SizedBox(height: 16),
              
              // Technical Indicators
              if (_indicators != null) _buildIndicatorsCard(),
              
              SizedBox(height: 16),
              
              // Backtest Results
              if (_backtestResults != null) _buildBacktestCard(),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildSignalCard() {
    final signal = _signals!['overall_signal'];
    final confidence = (_signals!['confidence'] as num).toDouble();
    final signals = _signals!['signals'] as List;
    
    Color signalColor;
    IconData signalIcon;
    
    switch (signal) {
      case 'BUY':
        signalColor = AppTheme.primaryGreen;
        signalIcon = Icons.arrow_upward;
        break;
      case 'SELL':
        signalColor = AppTheme.primaryRed;
        signalIcon = Icons.arrow_downward;
        break;
      default:
        signalColor = Colors.orange;
        signalIcon = Icons.remove;
    }
    
    return Card(
      color: AppTheme.cardDark,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.analytics, color: AppTheme.primaryBlue),
                SizedBox(width: 8),
                Text(
                  'Trading Signal',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            SizedBox(height: 16),
            
            Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: signalColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: signalColor, width: 2),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Icon(signalIcon, color: signalColor, size: 32),
                      SizedBox(width: 12),
                      Text(
                        signal,
                        style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: signalColor,
                        ),
                      ),
                    ],
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                        'Confidence',
                        style: TextStyle(color: Colors.grey, fontSize: 12),
                      ),
                      Text(
                        '${(confidence * 100).toStringAsFixed(0)}%',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            
            if (signals.isNotEmpty) ...[
              SizedBox(height: 16),
              Text(
                'Signal Components',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 8),
              ...signals.map((s) => Padding(
                padding: EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(s['indicator'], style: TextStyle(color: Colors.grey)),
                    Text(
                      '${s['signal']} - ${s['reason']}',
                      style: TextStyle(
                        color: s['signal'] == 'BUY'
                            ? AppTheme.primaryGreen
                            : s['signal'] == 'SELL'
                                ? AppTheme.primaryRed
                                : Colors.orange,
                      ),
                    ),
                  ],
                ),
              )).toList(),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildIndicatorsCard() {
    return Card(
      color: AppTheme.cardDark,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Technical Indicators',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            
            // RSI
            _buildIndicatorRow(
              'RSI',
              _indicators!.rsi.toStringAsFixed(2),
              _indicators!.rsi < 30
                  ? AppTheme.primaryGreen
                  : _indicators!.rsi > 70
                      ? AppTheme.primaryRed
                      : Colors.grey,
            ),
            
            // MACD
            _buildIndicatorRow(
              'MACD',
              _indicators!.macd['histogram']!.toStringAsFixed(4),
              _indicators!.macd['histogram']! > 0
                  ? AppTheme.primaryGreen
                  : AppTheme.primaryRed,
            ),
            
            // Bollinger Bands
            _buildIndicatorRow(
              'BB Width',
              (_indicators!.bollingerBands['bandwidth']! * 100).toStringAsFixed(2) + '%',
              Colors.grey,
            ),
            
            // Moving Averages
            _buildIndicatorRow(
              'SMA 20',
              '\$${_indicators!.sma20.toStringAsFixed(2)}',
              Colors.grey,
            ),
            
            if (_indicators!.sma50 > 0)
              _buildIndicatorRow(
                'SMA 50',
                '\$${_indicators!.sma50.toStringAsFixed(2)}',
                Colors.grey,
              ),
            
            if (_indicators!.sma200 > 0)
              _buildIndicatorRow(
                'SMA 200',
                '\$${_indicators!.sma200.toStringAsFixed(2)}',
                Colors.grey,
              ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildIndicatorRow(String name, String value, Color color) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(name, style: TextStyle(color: Colors.grey)),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildBacktestCard() {
    final results = _backtestResults!['results'] as List;
    
    return Card(
      color: AppTheme.cardDark,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Backtest Results (1 Year)',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            
            ...results.map((result) {
              final totalReturn = (result['total_return'] * 100).toStringAsFixed(2);
              final winRate = (result['win_rate'] * 100).toStringAsFixed(0);
              final sharpeRatio = result['sharpe_ratio'].toStringAsFixed(2);
              
              return Container(
                margin: EdgeInsets.only(bottom: 12),
                padding: EdgeInsets.all(12),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.withOpacity(0.3)),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          result['strategy'],
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        Text(
                          '$totalReturn%',
                          style: TextStyle(
                            color: result['total_return'] > 0
                                ? AppTheme.primaryGreen
                                : AppTheme.primaryRed,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Win Rate: $winRate%', style: TextStyle(fontSize: 12, color: Colors.grey)),
                        Text('Sharpe: $sharpeRatio', style: TextStyle(fontSize: 12, color: Colors.grey)),
                        Text('Trades: ${result['total_trades']}', style: TextStyle(fontSize: 12, color: Colors.grey)),
                      ],
                    ),
                  ],
                ),
              );
            }).toList(),
            
            if (_backtestResults!['best_strategy'] != null) ...[
              SizedBox(height: 8),
              Container(
                padding: EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppTheme.primaryBlue.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(Icons.star, color: AppTheme.primaryBlue, size: 16),
                    SizedBox(width: 8),
                    Text(
                      'Best Strategy: ${_backtestResults!['best_strategy']}',
                      style: TextStyle(color: AppTheme.primaryBlue),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// Additional Screens (Watchlist, Search, Settings, Position Detail)
// These would be implemented similarly with full functionality
// For brevity, I'm including the structure:

class WatchlistScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final watchlistProvider = Provider.of<WatchlistProvider>(context);
    
    return Scaffold(
      appBar: AppBar(title: Text('Watchlist')),
      body: Center(child: Text('Watchlist Screen')),
    );
  }
}

class SearchScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Search Stocks')),
      body: Center(child: Text('Search Screen')),
    );
  }
}

class SettingsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Settings')),
      body: ListView(
        children: [
          ListTile(
            leading: Icon(Icons.person),
            title: Text('Profile'),
            trailing: Icon(Icons.arrow_forward_ios),
          ),
          ListTile(
            leading: Icon(Icons.notifications),
            title: Text('Notifications'),
            trailing: Switch(value: true, onChanged: (v) {}),
          ),
          ListTile(
            leading: Icon(Icons.logout),
            title: Text('Sign Out'),
            onTap: () {
              Provider.of<AuthProvider>(context, listen: false).signOut();
            },
          ),
        ],
      ),
    );
  }
}

class PositionDetailScreen extends StatelessWidget {
  final Position position;
  
  PositionDetailScreen({required this.position});
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(position.symbol)),
      body: Center(child: Text('Position Details')),
    );
  }
}